include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
    NAME dr_libs
    GITHUB_REPOSITORY mackron/dr_libs
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

message(STATUS "RTNeural -- Configuring example: process_wav_torch")
add_executable(process_wav_torch process_wav_torch.cpp)
target_include_directories(process_wav_torch PRIVATE ${CMAKE_SOURCE_DIR} ${dr_libs_SOURCE_DIR})
target_link_libraries(process_wav_torch PRIVATE RTNeural)
target_compile_features(process_wav_torch PRIVATE cxx_std_17)

add_custom_command(TARGET process_wav_torch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:process_wav_torch> ${EXAMPLES_OUT_DIR}/process_wav_torch)
