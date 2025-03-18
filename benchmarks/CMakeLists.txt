# The list of benchs
file(GLOB_RECURSE all_benchs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

include(FetchContent)


set(BAOBZI_BUILD_STATIC ON)
set(BAOBZI_CPU_DISPATCH OFF)
# Fetch the baobzi library
FetchContent_Declare(
        baobzi
        GIT_REPOSITORY https://github.com/flatironinstitute/baobzi.git
        GIT_TAG main  # You can specify a specific commit, branch, or tag
)
FetchContent_MakeAvailable(baobzi)
unset(BAOBZI_BUILD_STATIC)
unset(BAOBZI_CPU_DISPATCH)


foreach(bench ${all_benchs})
  get_filename_component(bench_name ${bench} NAME_WE)
  get_filename_component(bench_dir ${bench} DIRECTORY)
  add_executable(${bench_name} ${bench})
  target_link_libraries(${bench_name} triqs triqs_warnings benchmark_main baobzi)
  set_property(TARGET ${bench_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${bench_dir})
  set_target_properties(${bench_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE})
  #add_bench(NAME ${bench_name} COMMAND ${bench_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${bench_dir})

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    set_target_properties(${bench_name} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  endif()
  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    add_custom_command(
      TARGET ${bench_name}
      COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=warning,style,performance,portability
      --std=c++20
      --template=gcc
      --verbose
      --force
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${bench}
    )
  endif()
endforeach()
