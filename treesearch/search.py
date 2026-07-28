import os
import pickle
import random
import shutil
from pathlib import Path

from anytree import PreOrderIter

from config import CONFIG_PATH, Config
from treesearch.interpreter import Interpreter
from treesearch.minimal_agent import MinimalAgent
from treesearch.node import Node
from treesearch.type_checker import TypeChecker
from utils.log import _ROOT_LOGGER
from utils.path import mkdir
from utils.statistics_tracker import get_statistics_tracker
from viz import render_trees

logger = _ROOT_LOGGER.getChild("treesearch")
statistics_tracker = get_statistics_tracker()


class TreeSearch:
    def __init__(self, user_request: str, config: Config) -> None:
        self._user_request = user_request
        self._config = config
        self._draft_nodes: list[Node] = []
        self._out_dir = mkdir(Path(config.out_dir))
        workspace_pth = mkdir(self._out_dir / "workspace").resolve()
        self._workspace = str(workspace_pth)
        self._checkpoint_dir = mkdir(self._out_dir / "checkpoint")

        shutil.copy(CONFIG_PATH, self._out_dir)

        self._prototype_agent = MinimalAgent(
            self._stage_task_desc("prototype"),
            self._config,
            evaluation_metrics=self._config.agent.evaluation_metrics,
            stage_name="prototype",
        )
        self._interpreter = Interpreter(self._workspace, self._config.exec.timeout)
        self._type_checker = TypeChecker(self._workspace)

    async def _async_init(self):
        await self._prototype_agent._async_init()

    @property
    def all_nodes(self):
        return [n for root in self._draft_nodes for n in PreOrderIter(root)]

    @property
    def good_nodes(self):
        return list(filter(lambda n: not n.is_buggy, self.all_nodes))

    @property
    def buggy_nodes(self):
        return list(filter(lambda n: n.is_buggy, self.all_nodes))

    @property
    def best_good_node(self):
        good_nodes = self.good_nodes
        good_nodes.sort(key=lambda n: (n.score.score, n.ctime), reverse=True)
        return good_nodes[0]

    @property
    def best_buggy_node(self):
        buggy_nodes = self.buggy_nodes
        buggy_nodes.sort(key=lambda n: (n.score.score, n.ctime), reverse=True)
        return buggy_nodes[0]

    def select_next_node(self) -> Node:
        if (
            len(self.buggy_nodes) > 0
            and random.random() < self._config.treesearch.debug_prob
            or len(self.good_nodes) == 0
        ):
            if random.random() < self._config.treesearch.epsilon:
                logger.info("Selecting random buggy node for debugging...")
                nodes = self.buggy_nodes
                weights = [1 / (len(n.children) + 1) for n in nodes]
                return random.choices(nodes, weights=weights, k=1)[0]
            else:
                logger.info("Selecting best buggy node for debugging...")
                return max(
                    self.buggy_nodes,
                    key=lambda n: n.score.score * (1 / (len(n.children) + 1)),
                )

        if random.random() < self._config.treesearch.epsilon:
            nodes = self.good_nodes
            weights = [1 / (len(n.children) + 1) for n in nodes]
            return random.choices(nodes, weights=weights, k=1)[0]
        else:
            return max(
                self.good_nodes,
                key=lambda n: n.score.score * (1 / (len(n.children) + 1)),
            )

    async def run(self):
        logger.info("Starting tree search...")
        # Step 1: Generate draft nodes:
        for i in range(self._config.treesearch.num_draft_nodes):
            logger.info(
                f"Generating draft node {i + 1}/{self._config.treesearch.num_draft_nodes}"
            )
            draft_node = await self._prototype_agent._draft()
            await self.exec_node(draft_node, self._prototype_agent)
            self._draft_nodes.append(draft_node)
            statistics_tracker.add_node(draft_node)

        best_node: Node | None = None
        for i in range(self._config.treesearch.max_iterations):
            logger.info(
                f"Treesearch iteration {i + 1}/{self._config.treesearch.max_iterations}"
            )
            parent_node = self.select_next_node()

            if parent_node.is_buggy:
                child_node = await self._prototype_agent._debug(parent_node)
            else:
                child_node = await self._prototype_agent._improve(parent_node)

            await self.exec_node(child_node, self._prototype_agent)
            statistics_tracker.add_node(child_node)

            if child_node.score.is_satisfactory:
                logger.info(
                    "Found satisfactory prototype node; proceeding to final refinement."
                )
                best_node = child_node
                break

        self.save()

        if best_node is None:
            logger.warning(
                "Found no satisfactory prototype node; Using best node instead..."
            )

            if len(self.good_nodes) == 0:
                logger.warning("No good nodes found; Using best buggy node...")
                best_node = self.best_buggy_node
            else:
                best_node = self.best_good_node

        if best_node is None:
            raise RuntimeError("No node available for final refinement.")

        # Step 2: Refinement loop with final requirements
        refined_node, final_agent = await self._refine_best_node(best_node)
        await self.finalize_search(result_node=refined_node, agent=final_agent)

    async def exec_node(self, node: Node, agent: MinimalAgent) -> Node:
        # Type checking refinement loop
        current_code = node.code

        if self._config.exec.enable_type_checking:
            max_type_check_attempts = self._config.exec.max_type_check_attempts

            for attempt in range(1, max_type_check_attempts + 1):
                node.type_check_attempts = attempt
                logger.info(
                    f"Type checking code (attempt {attempt}/{max_type_check_attempts})..."
                )

                type_check_result = self._type_checker.check_code(current_code)

                if not type_check_result.has_errors:
                    logger.info("Type checking passed!")
                    node.type_check_passed = True
                    break

                logger.warning(
                    f"Type checking found {type_check_result.error_count} error(s) "
                    f"(attempt {attempt}/{max_type_check_attempts})"
                )
                node.type_check_results.append(type_check_result)

                if attempt == max_type_check_attempts:
                    logger.warning(
                        "Max type checking attempts reached. Proceeding with execution despite type errors."
                    )
                    node.type_check_passed = False
                    break

                logger.info("Attempting to fix type errors using LLM...")
                try:
                    fixed_code = await agent._fix_type_errors(
                        current_code, type_check_result.format_errors_for_llm()
                    )
                    current_code = fixed_code
                except Exception as e:
                    logger.error(f"Failed to fix type errors: {e}")
                    node.type_check_passed = False
                    break
        else:
            logger.info(
                "Type checking is disabled. Enable it in the config to refine code before execution."
            )
            node.type_check_passed = None  # type: ignore

        # Always sync node.code with current_code so that what we execute
        # matches what the agent sees later
        node.code = current_code

        exec_result = self._interpreter.run(current_code)
        logger.debug(exec_result)

        node_dir = mkdir(self._checkpoint_dir / node.id)
        (node_dir / "code.py").write_text(node.code)
        (node_dir / "out.log").write_text("".join(exec_result.term_out))
        (node_dir / "exec_result.pkl").write_bytes(pickle.dumps(exec_result))

        # Move all generated files from the workspace to checkpoint for this node
        workspace_dir = Path(self._workspace)
        working_dir = workspace_dir / "working"

        # Collect files from workspace (excluding runfile.py and working dir)
        generated_files = [
            item
            for item in workspace_dir.iterdir()
            if item.name not in ("runfile.py", "working")
            and not item.name.startswith(".")
        ]

        # Also collect files from working subdirectory if it exists
        if working_dir.exists():
            generated_files.extend([
                item for item in working_dir.iterdir()
                if item.suffix.lower() not in ignored_extensions
            ])

        # Keep only relevant files via whitelist
        if self._config.exec.keep_only_relevant_files:
            logger.info("Keeping only relevant files.")
            keep = []
            for item in generated_files:
                if item.suffix.lower() in (".png", ".jpeg", ".jpg", ".json", ".csv"):
                    logger.debug(f"Keeping {item.name}")
                    keep.append(item)
                else:
                    logger.debug(f"Removing {item.name}")
                    if item.is_dir():
                        shutil.rmtree(str(item))
                    else:
                        os.remove(str(item))

            generated_files = keep
        else:
            logger.info("Keeping all files.")

        if generated_files:
            generated_dir = mkdir(node_dir / "generated")
            for item in generated_files:
                try:
                    shutil.move(str(item), str(generated_dir / item.name))
                    logger.info(f"Moved {item.name} to checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to move {item.name}: {e}")

        await agent.score_code(node, exec_result)
        return node

    async def finalize_search(self, result_node: Node, agent: MinimalAgent):
        self._interpreter.cleanup_session()
        logger.info(f"Finalizing search with node: {result_node.id}")
        logger.info("Final response:")
        summary = await agent._summarize(self._user_request, result_node)
        summary_path = self._out_dir / "summary.md"
        summary_path.write_text(summary, encoding="utf-8")
        logger.info(f"Wrote markdown summary to: {summary_path}")
        print(summary)

    @property
    def _task_desc(self) -> str:
        task_desc = """ You are an expert recommender systems research assistant who is looking to help the user with their requests.
                    The user has some idea and you want to conduct creative experiments to gain scientific insights.
                    Your aim is to run experiments to gather sufficient results to report back to the user.
                    The idea is:\n
                    """
        task_desc += self._user_request
        return task_desc

    def _stage_task_desc(self, stage: str) -> str:
        if stage == "prototype":
            return (
                "You are in the PROTOTYPE stage."
                " Your goal is to create a minimal pilot that demonstrates the end-to-end pipeline."
                " Follow ONLY the prototype requirements: exactly one dataset, exactly one algorithm, minimal metrics, and at least one plot."
                " Do NOT attempt to satisfy the full user request in this stage."
                "\n\n" + self._task_desc
            )
        if stage == "final":
            return (
                "You are in the FINAL stage."
                " You will be given a working prototype script from a previous node."
                " Your job is to INCREMENTALLY extend it to satisfy the full user request (e.g., add datasets, algorithms, metrics, plots)."
                " Preserve the existing code structure and outputs as much as possible."
                " Do NOT rewrite the script from scratch unless it is strictly required for correctness."
                "\n\n" + self._task_desc
            )
        return self._task_desc

    def save(self):
        logger.info("Generating tree visualization...")
        tree_render_dir = mkdir(self._out_dir / "tree_render")
        render_trees(self._draft_nodes, tree_render_dir)

        with open(self._out_dir / "save.pkl", "wb") as f:
            logger.info(f"SAVING {len(self._draft_nodes)}.....")
            pickle.dump(self._draft_nodes, f)

    async def _refine_best_node(self, best_node: Node) -> tuple[Node, MinimalAgent]:
        logger.info("Starting refinement loop with final requirements...")

        refinement_base_id = best_node.id
        if "_iteration" in refinement_base_id:
            refinement_base_id = refinement_base_id.split("_iteration", 1)[0]
        if refinement_base_id.endswith("_seed"):
            refinement_base_id = refinement_base_id[: -len("_seed")]

        final_agent = MinimalAgent(
            self._stage_task_desc("final"),
            self._config,
            evaluation_metrics=self._config.agent.evaluation_metrics,
            stage_name="final",
            selected_datasets=self._prototype_agent.selected_datasets,
        )
        await final_agent._async_init()

        # Seed node: re-run best prototype code under final requirements
        seed_node = final_agent._new_node(
            best_node.plan, best_node.code, parent=best_node
        )
        seed_node.id = f"{refinement_base_id}_seed"
        await self.exec_node(seed_node, final_agent)
        statistics_tracker.add_node(seed_node)

        current_best = seed_node
        for i in range(self._config.treesearch.refinement_iterations):
            logger.info(
                f"Refinement iteration {i + 1}/{self._config.treesearch.refinement_iterations}"
            )

            parent_node = current_best
            if parent_node.is_buggy:
                child_node = await final_agent._debug(parent_node)
            else:
                child_node = await final_agent._improve(parent_node)

            child_node.id = f"{refinement_base_id}_iteration{i + 1}"

            await self.exec_node(child_node, final_agent)
            statistics_tracker.add_node(child_node)

            if child_node.score.is_satisfactory:
                logger.info("Found satisfactory node in refinement loop.")
                return child_node, final_agent

            if child_node.score.score >= current_best.score.score:
                current_best = child_node

        logger.warning(
            "Refinement loop ended without full satisfaction; using best refined node."
        )
        return current_best, final_agent
