include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
    NAME dr_libs
    GITHUB_REPOSITORY mackron/dr_libs
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

message(STATUS "RTNeural -- Configuring example: process_wav_torch_param2")
add_executable(process_wav_torch_param2 process_wav_torch_param2.cpp)
target_include_directories(process_wav_torch_param2 PRIVATE ${CMAKE_SOURCE_DIR} ${dr_libs_SOURCE_DIR})
target_link_libraries(process_wav_torch_param2 PRIVATE RTNeural)
target_compile_features(process_wav_torch_param2 PRIVATE cxx_std_17)

add_custom_command(TARGET process_wav_torch_param2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:process_wav_torch_param2> ${EXAMPLES_OUT_DIR}/process_wav_torch_param2)