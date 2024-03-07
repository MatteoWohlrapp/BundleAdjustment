cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(ba_project)

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.12 REQUIRED) 

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# Check if the build type is Debug
if (CMAKE_BUILD_TYPE MATCHES Debug)
    # Add -fsanitize=address to the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    # Link the sanitizer library when using GCC
    if (CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    endif()
endif()

set(HEADER_FILES 
    src/utils/Eigen.h 
    src/visualization/SimpleMesh.h 
    src/model/PointCloud.h 
    src/data/VirtualSensor.h 
    src/data/FreeImageHelper.h
    src/ba/FeatureProcessor.h
    src/model/Frame.h
    src/model/MapPoint.h
    src/model/SceneMap.h
    src/ba/Optimizer.h
    src/ba/Initializer.h
    src/ba/SfMHelper.h
    src/metrics/ReconstructionError.h
    src/ba/BundleAdjustment.h
    src/visualization/Visualizer.h
)

set(SOURCE_FILES 
    src/data/FreeImageHelper.cpp
    src/ba/FeatureProcessor.cpp
    src/model/Frame.cpp
    src/model/MapPoint.cpp
    src/model/SceneMap.cpp
    src/ba/Initializer.cpp
    src/ba/SfMHelper.cpp
    src/ba/Optimizer.cpp
    src/metrics/ReconstructionError.cpp
    src/visualization/SimpleMesh.cpp 
    src/ba/BundleAdjustment.cpp
    src/visualization/Visualizer.cpp
)
link_directories(${PCL_LIBRARY_DIR} ${FreeImage_LIBRARY_DIR})
add_executable(ba_project src/main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ba_project ceres ${PCL_LIBRARIES} freeimage Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(ba_project PUBLIC ${PCL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})


# On Windows copy dll to executable folder
if(WIN32)
   # Visual Studio properties
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ba_project)
    set_property(TARGET ba_project PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # copies all files (.dll and .exe) to the build folder from "3dsmc_project\libs\PCL-1.12.0\bin" and "3dsmc_project\libs\PCL-1.12.0\3rdParty\VTK\bin"
    add_custom_command(
        TARGET ba_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${LIBRARY_DIR}/PCL-1.12.0/bin/"
                ${CMAKE_BINARY_DIR})
    add_custom_command(
        TARGET ba_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${LIBRARY_DIR}/PCL-1.12.0/3rdParty/VTK/bin/"
                ${CMAKE_BINARY_DIR})

    # On Windows copy dll to executable folder
    add_custom_command(TARGET ba_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET ba_project POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            ${GLOG_DLL_PATH_DEBUG}                       
            ${CMAKE_BINARY_DIR}
	)

    add_custom_command(TARGET ba_project POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different   
            $<IF:$<CONFIG:Release>,"${OpenCV_DIR}/x64/vc17/bin/opencv_world481.dll","${OpenCV_DIR}/x64/vc17/bin/opencv_world481d.dll">
            ${CMAKE_BINARY_DIR}
	)

    add_custom_command(TARGET ba_project POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different   
            "${OPENNI2_INCLUDE_DIR}/../Tools/OpenNI2.dll"
            ${CMAKE_BINARY_DIR}
	)

endif(WIN32)

# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# include(googletest)