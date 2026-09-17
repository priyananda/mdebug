# About the project - Model Debugger

In this repo, we are going to develop a "Model Inference Debugger", which aims to build an experience similar to debudding code, but for LLM text generation. The primary uses of a code debugger are:
*  Control the execution of the code (setting breakpoints, skipping checkpoints)
*  Visualize intermediate state information (variables, stack, registers)

So what's the point of debugging LLM text generation? There are a few use cases we can think of:
*  Understand how inference works and looking at how data flows
*  Change the parameters or behavior on the fly for adhoc analysis.
*  (Most useful) Visualize intermdiate state in a useful way (KV cache, token vectors)

The visualization is the most interesting bit, since intermediate state for an LLM is usally complex (high dimensional vectors, large matrices)

# Architectural guidelines

We want to implement this as a client-server application.

*  The client is an Angular single page application which visualizes execution and intermdiate state.
*  The server is a pyton web server which includes the inference engine.

The client and server will communicate via HTTP calls and WebSockets.

# About code structure

The folder structure follows the overall architecture
*  `server/` holds the server-side code
*  `client/` holds the client-side code

# Coding guidelines

*   Follow idiomatic conventions for the language (python, typescript)
*   This is for demonstration purposes, not production code. Should be reliable, but not necessarily production grade

# Deployment

*   The client (html, generate javascript files) will be hosted on github.io
*   The server (python files and dependencies) needs to be packaged as a docker container and deployed to Google Cloud.