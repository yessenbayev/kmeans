include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)



#######################################
#######################################
file(GLOB SOURCES  *.h *.hpp *.cu *.cuh *.cpp )
source_group("Headers" FILES ${HEADERS})

cuda_add_executable( ece285kmeans-TriangleInequality
${SOURCES}
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G
)

set_target_properties( ece285kmeans-TriangleInequality
PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
"${CMAKE_SOURCE_DIR}/Src/ece285kmeans-Data")

target_link_libraries(
ece285kmeans-TriangleInequality

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries(
ece285kmeans-TriangleInequality
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)
