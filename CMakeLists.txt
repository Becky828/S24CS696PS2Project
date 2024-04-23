# CMakeList.txt : CMake project for CMakeS24CS696_PS2_p2, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CMakeS24CS696_PS2_p2")

# Add source to this project's executable.
add_executable (CMakeS24CS696_PS2_p2 "collaborative_filtering_template-3.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeS24CS696_PS2_p2 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

# Enable testing for this project
enable_testing()
include(CTest)

# Update the submodules here
include(UpdateSubmodules.cmake)

# Add subdirectories with code
add_subdirectory(tests)
add_subdirectory(external)

#Copying required files and folders
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ratings.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/very_abridged_Dataset.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/move-ratings/Dataset.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/move-ratings/Movie_Id_Titles.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
