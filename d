Mike.S (1):
      Added the Movielens dataset

Moritz Baumgart (36):
      chore: initial commit
      feat: copied and adapted required functionality from AISv2
      feat: add config with default values, add basic draft node generation structures, auto configure logging using rich.logging
      fix: add missing utility methods used in openai and anthropic backend
      feat: add basic draft loop generation, execution and evaluation
      fix: change wrong import path in minimal agent
      feat: add required llm and utility dependencies
      feat: add workspace to config
      feat: implement complete tree search selection strategy iteration loop
      feat: make satisfactory threshold configurable
      feat: downgraded to python 3.11 to install lenskit
      fix: fixed off by 1 error in progress logging
      fix: change all_nodes property to handle reference correctly
      chore: add vscode launch.json for debugging
      fix: add sklearn as dependency
      fix: config fields are now correctly accesed
      feat: render buggy state in node name
      feat: add viz.py to visualize treesearch trees saved as pickle
      feat: add files for running inside docker
      feat: save code requirements to output directory
      fix: set code workspace to be inside ./out for docker compatibility
      feat: add node score to name property of node
      fix: set tree search save path to be inside ./out for docker compatibility
      feat: add cli prompt for user request
      fix: make running with docker interactive for using the cli input
      feat: add best node summary at the end of treesearch
      feat: add PS1 to Dockerfile and start in interactive bash session instead of immediately running AutoRecLab
      feat: add startscript that also captures output to a file
      feat: added file logging handler and some more general logging
      merge: fix lost git history
      fix: removed unused config attribute
      fix: change tee log path to current dir so it does not run in file not exist error
      docs: add example to readme
      docs: add paper title to readme
      docs: add hint to develop branch to readme
      chore: bump version to 1.0.0

Philipp Meister (5):
      feat: added scoring function
      feat: added requirements-based scoring replacing the static scoring based on different quality aspects. Refactored the scoring so it uses function specs.
      docs: added content to README
      chore: initial commit
      chore: changed config default values

sinan-pourazari (2):
      refactor: removed regex parsing and introduce tool-call based code extraction
      fix: added plan_and_code_spec

