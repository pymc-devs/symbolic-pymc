# Docker FAQ

* Image can be built locally using the command `make docker` or the command
 `./scripts/container.sh --build` from the root `symbolic-pymc` directory

* After image is built an interactive bash session can be run
  `docker run -it symbolic-pymc bash`

* Command can be issued to the container such as linting and testing
  without interactive session
  * `docker run symbolic-pymc bash -c "pytest tests"`
