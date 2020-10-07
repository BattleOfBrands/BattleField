## Setting up the project


Help
`make help` or `make`

Running the project on local?

    1. Create python virtual environment and install dependencies
    `make env`
    
    2. Run the project
    `make run`

Built something?

    1. Run Unit test cases
    `make test_unit`
    
    2. Run Integration test cases
    `make test_integration`
    
Building docker image
    
    You will require docker to be installed on your system, if its not already present.

    1. Create and run docker image
    `make build_docker`
    
Everything seems to be working, how to deploy the project?

    1. Push the code to GitHub.
    2. Brainstrom on how we are going to use the service