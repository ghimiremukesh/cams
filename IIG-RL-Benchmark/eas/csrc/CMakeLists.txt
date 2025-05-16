add_library(traverser traverser.cpp averager.cpp cfr.cpp)
target_include_directories(traverser PUBLIC ${Boost_INCLUDE_DIRS})

add_executable(eas eas.cpp)
target_link_libraries(eas traverser)

python_add_library(pyeas MODULE bindings.cpp WITH_SOABI)
target_link_libraries(pyeas PRIVATE traverser pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(pyeas PRIVATE VERSION_INFO=${PROJECT_VERSION})

message(STATUS "SKBUILD_PROJECT_NAME: ${SKBUILD_PROJECT_NAME}")
# The install directory is the output (wheel) directory
install(TARGETS pyeas DESTINATION ${SKBUILD_PROJECT_NAME})

