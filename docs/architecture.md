# Architecture Overview

AutoRecLab is organized around a search-driven execution loop. The system generates candidate solutions, evaluates them, and iteratively refines the most promising ones.

## High-level execution flow

The runtime follows this sequence:

1. The CLI entry point in [main.py](../main.py) parses arguments and loads configuration.
2. The configuration layer in [config.py](../config.py) reads defaults from [config.toml](../config.toml) and environment variables prefixed with `ARL_`.
3. The tree-search orchestrator in [treesearch/search.py](../treesearch/search.py) creates a workspace and a search tree.
4. The agent in [treesearch/minimal_agent.py](../treesearch/minimal_agent.py) translates the research request into requirements, a plan, and executable code.
5. The interpreter in [treesearch/interpreter.py](../treesearch/interpreter.py) executes generated Python in a subprocess.
6. The type checker in [treesearch/type_checker.py](../treesearch/type_checker.py) optionally validates the code before execution.
7. Each result is stored as a node in the search tree, scored, and either improved or debugged further.
8. The final state is summarized and saved to disk.

## Main components

### 1. Entry point and CLI

The main entry point is [main.py](../main.py). It is responsible for:

- parsing CLI arguments
- initializing directories
- optionally listing datasets or models
- collecting the user request
- starting the search engine

The CLI supports both interactive prompt entry and non-interactive modes such as `--prompt` and `--prompt-file`.

### 2. Configuration

The configuration layer in [config.py](../config.py) provides a typed settings object based on Pydantic settings and Toml support.

It loads:

- default values from [config.toml](../config.toml)
- environment overrides such as `ARL_out_dir`
- nested configuration values using the `__` delimiter

This makes the runtime behavior tunable without editing code directly.

### 3. Tree search engine

The core orchestration logic lives in [treesearch/search.py](../treesearch/search.py). It manages:

- draft node creation
- node selection for debugging or improvement
- execution of each generated candidate
- persistence of checkpoints and artifacts
- final summarization

Its design is based on a tree of candidate solutions, where each node represents one attempt at solving the task.

### 4. Minimal agent

The agent implementation in [treesearch/minimal_agent.py](../treesearch/minimal_agent.py) acts as the reasoning layer.

It is responsible for:

- selecting relevant datasets
- deriving experiment requirements
- generating a plan and code
- debugging failing implementations
- improving existing solutions
- fixing static type issues
- summarizing final outcomes

The agent interacts with an LLM and uses documentation search tools to ground its responses in framework-specific guidance.

### 5. LLM and documentation retrieval

The LLM integration is handled by [treesearch/llm/query.py](../treesearch/llm/query.py). It wraps LangChain and OpenAI models and can attach tool-based capabilities.

The project uses a documentation search server in [treesearch/mcp/docs_search_server.py](../treesearch/mcp/docs_search_server.py) to retrieve relevant documentation passages. This helps the agent avoid guessing API usage and instead consult framework documentation.

### 6. Interpreter

The interpreter in [treesearch/interpreter.py](../treesearch/interpreter.py) executes generated code in a subprocess and captures:

- standard output and error output
- whether an exception occurred
- execution time

This isolation helps prevent broken experiments from corrupting the main runtime.

### 7. Type checker

The type checker in [treesearch/type_checker.py](../treesearch/type_checker.py) runs the `ty` checker over generated Python code.

When enabled, it can:

- catch obvious static typing issues before execution
- provide feedback to the agent for fix-up cycles
- reduce the chance of runtime-only errors

### 8. Node model

The node abstraction in [treesearch/node.py](../treesearch/node.py) encapsulates a single candidate solution. Each node stores:

- the generated plan
- the corresponding code
- execution output and time
- scoring and feedback
- whether the node was considered buggy
- type-checking results

This structure allows the search engine to reason over the quality and history of each attempt.

### 9. Visualization

The utility in [viz.py](../viz.py) renders the search tree as PNG, SVG, or PDF files. This is useful when you want to understand which branches were explored and which produced the strongest outcomes.

## Data flow in one sentence

A user prompt enters the CLI, is transformed into requirements and candidate code by the agent, executed and scored by the interpreter, and then either improved or debugged further through the tree-search loop.
