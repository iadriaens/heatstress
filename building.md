# Install prerequisites

## Install WSL
WSL is Windows Subsystem for Linux, which allows you to run a Linux kernel from
Windows. It is a prerequisite for installing Docker on Windows.

1. Open a PowerShell prompt in administrator mode (search for "PowerShell" in
   the start menu, right click "Run as administrator").
2. Run the command `wsl --install`.

## Install Docker desktop
Docker is software to run "containers": effectively isolated operating system
environments in which you install precisely the libraries or software that you
need for a single task. Containers are executed from "images", which is a sort
of blueprint for what is installed in the container (both operating system and
tools and libraries). Containers usually run Linux distributions.

In our case, we will use a container that contains the development tools for the
MoveSense. This allows us to just run the container and start working, without
having to install an entire suite of tools and libraries.

Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
and run the installer. This will install the Docker engine (which allows you to
run the actual containers) and a GUI tool for interacting with containers.

## Install the MoveSense Showcase App
This app allows you to install firmware on the MoveSense sensor via your phone's
Bluetooth connection. For Apple devices, it is available on the app store,
search for "MoveSense Showcase".

Unfortunately, it is not available in the Play store for Android devices.
Instead, you have to manually install a `.apk`-file. Follow these steps:

1. On your phone, download `ShowCaseApp-<version>.apk`;
2. Open it; your phone will probably show a warning that installing unknown apps
   is not allowed. Follow the settings link provided to allow it.
3. Press install and verify that the app works.

Note that the app needs "Location" permissions to be able to use the Bluetooth
connection.

# Prepare and test the MoveSense development environment
We will mostly interact with Docker from a PowerShell prompt. All of these
actions can also be performed from Docker Desktop, but having copyable commands
is convenient for this type of step-by-step manual.

*IMPORTANT*: Before trying to run Docker commands, start Docker desktop to
ensure that the "Docker daemon" service is running. Otherwise, you will get
errors stating the latter.

## Pull the MoveSense development image
Docker has a public repository of images, which MoveSense has submitted its
development container to. The command `docker pull` allows you to download
images from this repository to your machine, which will allow you to run
containers from this image.

Open a PowerShell window (no need for administrator mode) and run the following
command:
```PS
docker pull movesense/sensor-build-env:2.2
```
This will download the MoveSense development environment of the specified
version. It may take a while, you can continue to the next section while you
wait.

*IMPORTANT*: Do not be tempted to pull the `latest` image, as this is actually
not at all the latest image available...

## Clone the MoveSense device library
MoveSense provides a library to interact with the sensor and Bluetooth hardware
in the sensor. It is hosted on a public
[`git` repository](https://bitbucket.org/movesense/movesense-device-lib)

First, install [git for Windows](https://gitforwindows.org/) if you do not yet
have `git` on your machine.

Once installed, from a PowerShell window, navigate to the directory where you
want the library to be stored, then clone it with `git`:
```PS
git clone --branch release_2.2.1 https://bitbucket.org/movesense/movesense-device-lib.git
```
The option `--branch release_2.2.1` checks out the repository at that specific
"tag", which marks a specific release of the library.

*IMPORTANT*: Make sure the release that you clone matches the version of the
Docker image that you `pull`ed.

You can see which tags are available on the BitBucket page by clicking the
"master"-button below the library name and navigating to "Tags". Newer releases
will become available periodically, and will be tagged in this repository.

Also note that the URL in the `git`-command is differen from the BitBucket page;
it can be obtained via the "Clone" button on BitBucket. This also takes a while
(the repository contains many samples), but you have to wait for it to complete
to continue.

## Run the MoveSense development container
When everything is done `pull`ing and `clone`ing, you are ready to run the
development container. Use the following command, replacing the part between <>
with the full path to the directory where you cloned the MoveSense device
library:
```PS
docker run -it --rm -v <DIRECTORY WHERE YOU CLONED movesense-device-lib>:/movesense movesense/sensor-build-env:2.2
```
The `-it` argument indicates that you want to run a container that you want to
interact (`-i`) with via a terminal (`-t`). The `--rm` argument makes sure that
the container is removed automatically after it is stopped; we don't need to
keep it since we will just start a fresh one the next time we are developing.
Finally, `-v` "mounts" a directory on your Windows machine to the `/movesense`
directory inside the container so it can be accessed from inside the container.

## Building an example
Once "inside" the container, your prompt should change to `root@<something>:/#`,
you are now in a Linux environment prepared for building MoveSense applications.

Let's try building the "Hello World" example. First, navigate to the appropriate
sample directory:
```bash
cd /movesense/samples/hello_world_app
```
Note that the `/movesense` directory corresponds to the `movesense-device-lib`
directory in your Windows environment that you passed to the `docker run`
command.

Then, we use CMake to prepare the build system to compile the firmware. CMake is
a tool that allows you to specify a build for different platforms: which source
files and dependencies to use, which compiler settings, etc. CMake then uses
this general specification to generate build files specific for your platform.
These build files can then be used to compile the software. Run the following
command:
```bash
cmake -G Ninja -DMOVESENSE_CORE_LIBRARY=/movesense/MovesenseCoreLib/ -DCMAKE_TOOLCHAIN_FILE=/movesense/MovesenseCoreLib/toolchain/gcc-nrf52.cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
```
CMake will generate build files for the Ninja build system based on `-G Ninja`.
The command sets three configuration parameters with `-D` flags:
`MOVESENSE_CORE_LIBRARY` which is the path to the MoveSense core library,
`CMAKE_TOOLCHAIN_FILE` which is the path to the compiler-related configuration
file, and `CMAKE_BUILD_TYPE` indicating that we want a "Release" (not "Debug")
build. Finally, we let it know that our sources are in the current directory
with `-S .`, and that we want our build files to end up in the `build/`
directory with `-B build`.

Note: if you're getting errors here about the wrong version of the compiler,
you're probably running the wrong Docker container. Make sure to explicitly add
a version tag when doing `docker pull`.

Finally, build the firmware package with the following command:
```bash
cmake --build build --target pkgs
```
This instructs CMake to execute the prepared build in the directory `build/`,
and compile "pkgs" (i.e. firmware packages) and its dependencies.

This should have created two `.zip` files (among other stuff) in the `build/`
directory. Note that you can access these files from your normal Windows
environment: the directory that you have been working in inside the container
corresponds to the directory you passed to the `docker run` command.

One `.zip`-file contains the firmware with the bootloader, the other without.
You only need to upload the version with the bootloader (`*_w_bootloader.zip`)
*once* for every new version of the MoveSense libraries. After updating the
bootloader once, you can use the firmware package without bootloader.

## Uploading the example firmware to the sensor
Without the MoveSense debugger, the easiest way to upload firmware to the sensor
is via your phone.

Make sure the appropriate firmware package (with or without bootloader) is on
your phone via a method of your choice (e.g. Google Drive).

Then open the MoveSense Showcase App and press the "DFU" button. Press "Select
file" and select the firmware package `.zip`-file. Then press "Select device"
and select the device to update. If no device shows up, make sure the sensor is
turned on. Then confirm that you want to update the firmware and wait for it to
complete.

To verify that the upload was indeed successful, press the "Movesense" button
from the app's start screen, connect to the sensor, press "App info", then press
"GET". The name should be "Sample HelloWorld".

# Building and uploading your own firmware

## Run the MoveSense development container
Like when compiling the example, we will be running a MoveSense development
container again. However, this time we will mount two directories in the
container: the MoveSense core libraries and your own application.
```PS
docker run -it --rm -v <DIRECTORY WHERE YOU CLONED movesense-device-lib>:/movesense-core -v <DIRECTORY OF YOUR APPLICATION:/movesense-app movesense/sensor-build-env:2.2
```

Once logged in to the container, you should see your own applications's files
in `/movesense-app`:
```bash
cd /movesense-app
ls
```

## Build your application
First, go to the directory that contains the source files of your application,
for example:
```bash
cd /movesense-app/activity_broadcast_app
ls
```
Verify that the file `CMakeLists.txt is present in this directory.

Similar to the example, we will first run CMake to configure the build, passing
it the appropriate path to the MoveSense core library and toolchain file:
```bash
cmake -G Ninja -DMOVESENSE_CORE_LIBRARY=/movesense-core/MovesenseCoreLib/ -DCMAKE_TOOLCHAIN_FILE=/movesense-core/MovesenseCoreLib/toolchain/gcc-nrf52.cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
```
Then build the firmware package:
```bash
cmake --build build --target pkgs
```

## Upload your application firmware to the sensor
Use the same procedure as outlined for the example to upload the firmware to the
sensor. Make sure to use the version with bootloader (`*_w_bootloader.zip`) when
uploading firmware built with a newer MoveSense library version than present on
the sensor.

Verify that your application was installed similar to the example application.
