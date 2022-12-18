# About ML-Agents package (`com.unity.ml-agents`)

The _ML-Agents_ package contains the primary C# SDK for the [Unity ML-Agents
Toolkit].

The package allows you to convert any Unity scene into a learning environment
and train character behaviors using a variety of machine learning algorithms.
Additionally, it allows you to embed these trained behaviors back into Unity
scenes to control your characters. More specifically, the package provides the
following core functionalities:

- Define Agents: entities, or characters, whose behavior will be learned. Agents
  are entities that generate observations (through sensors), take actions, and
  receive rewards from the environment.
- Define Behaviors: entities that specify how an agent should act. Multiple
  agents can share the same Behavior and a scene may have multiple Behaviors.
- Record demonstrations of an agent within the Editor. You can use
  demonstrations to help train a behavior for that agent.
- Embedding a trained behavior into the scene via the [Unity Inference Engine].
  Embedded behaviors allow you to switch an Agent between learning and
  inference.

Note that the _ML-Agents_ package does not contain the machine learning
algorithms for training behaviors. The _ML-Agents_ package only supports
instrumenting a Unity scene, setting it up for training, and then embedding the
trained model back into your Unity scene. The machine learning algorithms that
orchestrate training are part of the companion [Python package].

Note that we also provide an _ML-Agents Extensions_ package
(`com.unity.ml-agents.extensions`) that contains early/experimental features
that you may find useful. This package is only available from the [ML-Agents
GitHub repo].

## Package contents

The following table describes the package folder structure:

| **Location**           | **Description**                                                         |
| ---------------------- | ----------------------------------------------------------------------- |
| _Documentation~_       | Contains the documentation for the Unity package.                       |
| _Editor_               | Contains utilities for Editor windows and drawers.                      |
| _Plugins_              | Contains third-party DLLs.                                              |
| _Runtime_              | Contains core C# APIs for integrating ML-Agents into your Unity scene.  |
| _Runtime/Integrations_ | Contains utilities for integrating ML-Agents into specific game genres. |
| _Tests_                | Contains the unit tests for the package.                                |

<a name="Installation"></a>

## Installation

To install this _ML-Agents_ package, follow the instructions in the [Package
Manager documentation].

To install the companion Python package to enable training behaviors, follow the
[installation instructions] on our [GitHub repository].

### Advanced Installation
With the changes to Unity Package Manager in 2021, experimental packages will not show up in the package list and have to be installed manually. There are two recommended ways to install the package manually:

#### Github via Package Manager

In Unity 2019.4 or later, open the Package Manager, hit the "+" button, and select "Add package from git URL".

![Package Manager git URL](https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/docs/images/unity_package_manager_git_url.png)

In the dialog that appears, enter
 ```
git+https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents#release_17
```

You can also edit your project's `manifest.json` directly and add the following line to the `dependencies`
section:
```
"com.unity.ml-agents": "git+https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents#release_17",
```
See [Git dependencies](https://docs.unity3d.com/Manual/upm-git.html#subfolder) for more information. Note that this
may take several minutes to resolve the packages the first time that you add it.

#### Local Installation for Development

[Clone the repository](https://github.com/Unity-Technologies/ml-agents/tree/release_17_docs/docs/Installation.md#clone-the-ml-agents-toolkit-repository-optional) and follow the
[Local Installation for Development](https://github.com/Unity-Technologies/ml-agents/tree/release_17_docs/docs/Installation.md#advanced-local-installation-for-development-1)
directions.

## Requirements

This version of the Unity ML-Agents package is compatible with the following
versions of the Unity Editor:

- 2019.4 and later

## Known Limitations

### Training

Training is limited to the Unity Editor and Standalone builds on Windows, MacOS,
and Linux with the Mono scripting backend. Currently, training does not work
with the IL2CPP scripting backend. Your environment will default to inference
mode if training is not supported or is not currently running.

### Inference

Inference is executed via the
[Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.barracuda@latest/index.html).

**CPU**

All platforms supported.

**GPU**

All platforms supported except:

- WebGL and GLES 3/2 on Android / iPhone

**NOTE:** Mobile platform support includes:

- Vulkan for Android
- Metal for iOS.

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations
from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time.
The Academy also moves in time with FixedUpdate() rather than Update(), so game
behavior implemented in Update() may be out of sync with the agent decision
making. See [Execution Order of Event Functions] for more information.

You can control the frequency of Academy stepping by calling
`Academy.Instance.DisableAutomaticStepping()`, and then calling
`Academy.Instance.EnvironmentStep()`

### Unity Inference Engine Models

Currently, only models created with our trainers are supported for running
ML-Agents with a neural network behavior.

## Helpful links

If you are new to the Unity ML-Agents package, or have a question after reading
the documentation, you can checkout our [GitHub Repository], which also includes
a number of ways to [connect with us] including our [ML-Agents Forum].

In order to improve the developer experience for Unity ML-Agents Toolkit, we have added in-editor analytics.
Please refer to "Information that is passively collected by Unity" in the
[Unity Privacy Policy](https://unity3d.com/legal/privacy-policy).

[unity ML-Agents Toolkit]: https://github.com/Unity-Technologies/ml-agents
[unity inference engine]: https://docs.unity3d.com/Packages/com.unity.barracuda@latest/index.html
[package manager documentation]: https://docs.unity3d.com/Manual/upm-ui-install.html
[installation instructions]: https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/docs/Installation.md
[github repository]: https://github.com/Unity-Technologies/ml-agents
[python package]: https://github.com/Unity-Technologies/ml-agents
[execution order of event functions]: https://docs.unity3d.com/Manual/ExecutionOrder.html
[connect with us]: https://github.com/Unity-Technologies/ml-agents#community-and-feedback
[ml-agents forum]: https://forum.unity.com/forums/ml-agents.453/
[ML-Agents GitHub repo]: https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/com.unity.ml-agents.extensions
