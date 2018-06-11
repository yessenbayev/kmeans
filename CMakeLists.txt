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

cuda_add_executable( ece285kmeans
${SOURCES}
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G
)

set_target_properties( ece285kmeans
PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
"${CMAKE_SOURCE_DIR}/Src/ece285kmeans")

target_link_libraries(
ece285kmeans

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries(
ece285kmeans
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)

add_custom_command(TARGET ece285kmeans POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/freeglut.dll"  
$<TARGET_FILE_DIR:ece285kmeans>)          

add_custom_command(TARGET ece285kmeans POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/glew64.dll"  
$<TARGET_FILE_DIR:ece285kmeans>)   