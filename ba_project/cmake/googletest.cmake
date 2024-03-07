# Prevent this file from being included more than once
if(GOOGLETEST_INCLUDED)
    return()
endif()
set(GOOGLETEST_INCLUDED TRUE)

include(FetchContent)
FetchContent_Declare(
    googletest 
    GIT_REPOSITORY https://github.com/google/googletest.git
    #update release version frequently
    GIT_TAG release-1.11.0
) 
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED) 
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BUILD_DIR})
endif()

enable_testing()


file(GLOB_RECURSE TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/ReconstructionError_test.cc"
    # header don't need to be included but this might be necessary for some IDEs
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

set(TEST_SOURCE_FILES 
    src/ba/FeatureProcessor.cpp
    src/model/Frame.cpp
    src/model/MapPoint.cpp
    src/model/SceneMap.cpp
    src/ba/Initializer.cpp
    src/ba/SfMHelper.cpp
    src/ba/Optimizer.cpp 
    src/ba/Optimizer.h
    src/visualization/SimpleMesh.h
    src/data/VirtualSensor.h
    src/data/FreeImageHelper.h
    src/data/FreeImageHelper.cpp
    src/metrics/ReconstructionError.cpp
    src/metrics/ReconstructionError.h
)

add_executable(
    AllTests
    ${TEST_SRC}
    ${TEST_SOURCE_FILES}
)

target_link_libraries(AllTests gtest gmock gtest_main ceres ${PCL_LIBRARIES} freeimage Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(AllTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

include(GoogleTest)
