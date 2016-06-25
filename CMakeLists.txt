cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

################################################################################
# Qt
#find_package(Qt4 REQUIRED)
include(cmake/pcl_find_qt5.cmake)

# OpenGL
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# OpenNI
find_package(OpenNI REQUIRED)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
################################################################################
set(SUBSYS_NAME apps)
set(SUBSUBSYS_NAME in_hand_scanner)

set(SRCS
  src/help_window.cpp
  src/main.cpp
  src/main_window.cpp
  src/icp.cpp
  src/in_hand_scanner.cpp
  src/input_data_processing.cpp
  src/integration.cpp
  src/mesh_processing.cpp
  src/opengl_viewer.cpp
  src/visibility_confidence.cpp
  )

set(IMPL_INCS
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/impl/common_types.hpp"
  )

set(INCS
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/boost.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/common_types.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/eigen.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/icp.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/input_data_processing.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/integration.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/mesh_processing.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/utils.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/visibility_confidence.h"
  )

# Qt
set(MOC_IN_HAND_SCANNER_INC     "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/in_hand_scanner.h")
set(MOC_OPENGL_VIEWER_INC       "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/opengl_viewer.h")
set(MOC_OFFLINE_INTEGRATION_INC "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/offline_integration.h")

set(MOC_MAIN_WINDOW_INC "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/main_window.h")
set(MOC_HELP_WINDOW_INC "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/help_window.h")
set(UI_MAIN_WINDOW      src/main_window.ui)
set(UI_HELP_WINDOW      src/help_window.ui)

# Offline integration
set(OI_INCS
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/integration.h"
  "include/pcl/${SUBSYS_NAME}/${SUBSUBSYS_NAME}/visibility_confidence.h"
  )

set(OI_SRCS
  src/main_offline_integration.cpp
  src/integration.cpp
  src/offline_integration.cpp
  src/opengl_viewer.cpp
  src/visibility_confidence.cpp
  )

################################################################################
# Qt
# http://qtnode.net/wiki/Qt4_with_cmake
# http://qt-project.org/quarterly/view/using_cmake_to_build_qt_projects
set(QT_USE_QTOPENGL TRUE)
include("${QT_USE_FILE}")
qt4_wrap_cpp(MOC_IN_HAND_SCANNER_SRC     "${MOC_IN_HAND_SCANNER_INC}")
qt4_wrap_cpp(MOC_OPENGL_VIEWER_SRC       "${MOC_OPENGL_VIEWER_INC}" OPTIONS -DBOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
qt4_wrap_cpp(MOC_OFFLINE_INTEGRATION_SRC "${MOC_OFFLINE_INTEGRATION_INC}" OPTIONS -DBOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)

qt4_wrap_cpp(MOC_MAIN_WINDOW_SRC "${MOC_MAIN_WINDOW_INC}")
qt4_wrap_cpp(MOC_HELP_WINDOW_SRC "${MOC_HELP_WINDOW_INC}")
qt4_wrap_ui(UI_MAIN_WINDOW_INC   "${UI_MAIN_WINDOW}")
qt4_wrap_ui(UI_HELP_WINDOW_INC   "${UI_HELP_WINDOW}")

#list(APPEND ADDITIONAL_LIBS ${QT_LIBRARIES})
include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")

# In-hand scanner
list(APPEND INCS "${MOC_IN_HAND_SCANNER_INC}" "${MOC_OPENGL_VIEWER_INC}" "${MOC_MAIN_WINDOW_INC}" "${MOC_HELP_WINDOW_INC}" "${UI_MAIN_WINDOW_INC}" "${UI_HELP_WINDOW_INC}")
list(APPEND SRCS "${MOC_IN_HAND_SCANNER_SRC}" "${MOC_OPENGL_VIEWER_SRC}" "${MOC_MAIN_WINDOW_SRC}" "${MOC_HELP_WINDOW_SRC}")

set(EXE_NAME "pcl_${SUBSYS_NAME}_${SUBSUBSYS_NAME}")
add_executable(${EXE_NAME} ${SRCS} ${INCS} ${IMPL_INCS})
target_link_libraries(${EXE_NAME} ${PCL_LIBRARIES} ${OPENGL_LIBRARIES} ${QT_LIBRARIES})

# Offline integration
list(APPEND OI_INCS "${MOC_OPENGL_VIEWER_INC}" "${MOC_OFFLINE_INTEGRATION_INC}")
list(APPEND OI_SRCS "${MOC_OPENGL_VIEWER_SRC}" "${MOC_OFFLINE_INTEGRATION_SRC}")

ADD_EXECUTABLE(pcl_offline_integration ${OI_SRCS} ${OI_INCS})
target_link_libraries(pcl_offline_integration ${PCL_LIBRARIES} ${OPENGL_LIBRARIES} ${QT_LIBRARIES})
