# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/juan/Downloads/clion-2017.1.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/juan/Downloads/clion-2017.1.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/juan/Neuronal Network Project"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/juan/Neuronal Network Project/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/Neuronal_Network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Neuronal_Network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Neuronal_Network.dir/flags.make

CMakeFiles/Neuronal_Network.dir/test.c.o: CMakeFiles/Neuronal_Network.dir/flags.make
CMakeFiles/Neuronal_Network.dir/test.c.o: ../test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/juan/Neuronal Network Project/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Neuronal_Network.dir/test.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/Neuronal_Network.dir/test.c.o   -c "/home/juan/Neuronal Network Project/test.c"

CMakeFiles/Neuronal_Network.dir/test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Neuronal_Network.dir/test.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/juan/Neuronal Network Project/test.c" > CMakeFiles/Neuronal_Network.dir/test.c.i

CMakeFiles/Neuronal_Network.dir/test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Neuronal_Network.dir/test.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/juan/Neuronal Network Project/test.c" -o CMakeFiles/Neuronal_Network.dir/test.c.s

CMakeFiles/Neuronal_Network.dir/test.c.o.requires:

.PHONY : CMakeFiles/Neuronal_Network.dir/test.c.o.requires

CMakeFiles/Neuronal_Network.dir/test.c.o.provides: CMakeFiles/Neuronal_Network.dir/test.c.o.requires
	$(MAKE) -f CMakeFiles/Neuronal_Network.dir/build.make CMakeFiles/Neuronal_Network.dir/test.c.o.provides.build
.PHONY : CMakeFiles/Neuronal_Network.dir/test.c.o.provides

CMakeFiles/Neuronal_Network.dir/test.c.o.provides.build: CMakeFiles/Neuronal_Network.dir/test.c.o


# Object files for target Neuronal_Network
Neuronal_Network_OBJECTS = \
"CMakeFiles/Neuronal_Network.dir/test.c.o"

# External object files for target Neuronal_Network
Neuronal_Network_EXTERNAL_OBJECTS =

Neuronal_Network: CMakeFiles/Neuronal_Network.dir/test.c.o
Neuronal_Network: CMakeFiles/Neuronal_Network.dir/build.make
Neuronal_Network: CMakeFiles/Neuronal_Network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/juan/Neuronal Network Project/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable Neuronal_Network"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Neuronal_Network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Neuronal_Network.dir/build: Neuronal_Network

.PHONY : CMakeFiles/Neuronal_Network.dir/build

CMakeFiles/Neuronal_Network.dir/requires: CMakeFiles/Neuronal_Network.dir/test.c.o.requires

.PHONY : CMakeFiles/Neuronal_Network.dir/requires

CMakeFiles/Neuronal_Network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Neuronal_Network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Neuronal_Network.dir/clean

CMakeFiles/Neuronal_Network.dir/depend:
	cd "/home/juan/Neuronal Network Project/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/juan/Neuronal Network Project" "/home/juan/Neuronal Network Project" "/home/juan/Neuronal Network Project/cmake-build-debug" "/home/juan/Neuronal Network Project/cmake-build-debug" "/home/juan/Neuronal Network Project/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Neuronal_Network.dir/depend
