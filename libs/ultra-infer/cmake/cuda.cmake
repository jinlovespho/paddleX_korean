if(NOT WITH_GPU)
  return()
endif()

# This is to eliminate the CMP0104 warnings from cmake 3.18+.
# Instead of setting CUDA_ARCHITECTURES, we will set CMAKE_CUDA_FLAGS.
set(CMAKE_CUDA_ARCHITECTURES OFF)

if(BUILD_ON_JETSON)
  set(fd_known_gpu_archs "53 62 72")
  set(fd_known_gpu_archs10 "53 62 72")
else()
  message("Using New Release Strategy - All Arches Package")
  set(fd_known_gpu_archs "35 50 52 60 61 70 75 80 86")
  set(fd_known_gpu_archs10 "35 50 52 60 61 70 75")
  set(fd_known_gpu_archs11 "50 60 61 70 75 80")
endif()

######################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(
      WRITE ${cufile}
      ""
      "#include \"stdio.h\"\n"
      "#include \"cuda.h\"\n"
      "#include \"cuda_runtime.h\"\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(
      COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${cufile}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
      RESULT_VARIABLE nvcc_res
      OUTPUT_VARIABLE nvcc_out
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      # only keep the last line of nvcc_out
      string(REGEX REPLACE ";" "\\\\;" nvcc_out "${nvcc_out}")
      string(REGEX REPLACE "\n" ";" nvcc_out "${nvcc_out}")
      list(GET nvcc_out -1 nvcc_out)
      string(REPLACE "2.1" "2.1(2.0)" nvcc_out "${nvcc_out}")
      set(CUDA_gpu_detect_output
          ${nvcc_out}
          CACHE INTERNAL
                "Returned GPU architectures from detect_installed_gpus tool"
                FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(
      STATUS
        "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable}
        ${fd_known_gpu_archs}
        PARENT_SCOPE)
  else()
    set(${out_variable}
        ${CUDA_gpu_detect_output}
        PARENT_SCOPE)
  endif()
endfunction()

########################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   select_nvcc_arch_flags(out_variable)
function(select_nvcc_arch_flags out_variable)
  # List of arch names
  set(archs_names
      "Kepler"
      "Maxwell"
      "Pascal"
      "Volta"
      "Turing"
      "Ampere"
      "All"
      "Manual")
  set(archs_name_default "All")
  list(APPEND archs_names "Auto")

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME
      ${archs_name_default}
      CACHE STRING "Select target NVIDIA GPU architecture.")
  set_property(CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${archs_names})
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " archs_names "${archs_names}")
    message(
      FATAL_ERROR "Only ${archs_names} architectures names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN
        ${fd_known_gpu_archs}
        CACHE
          STRING
          "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported"
    )
    set(CUDA_ARCH_PTX
        ""
        CACHE
          STRING
          "Specify 'virtual' PTX architectures to build PTX intermediate code for"
    )
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    if(BUILD_ON_JETSON)
      set(cuda_arch_bin "53")
    else()
      set(cuda_arch_bin "50")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    if(BUILD_ON_JETSON)
      set(cuda_arch_bin "62")
    else()
      set(cuda_arch_bin "60 61")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    if(BUILD_ON_JETSON)
      set(cuda_arch_bin "72")
    else()
      set(cuda_arch_bin "70")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    set(cuda_arch_bin "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ampere")
    if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.1) # CUDA 11.0
      set(cuda_arch_bin "80")
    elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0) # CUDA 11.1+
      set(cuda_arch_bin "80 86")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(cuda_arch_bin ${fd_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    message(
      STATUS
        "WARNING: This is just a warning for publishing release.
      You are building GPU version without supporting different architectures.
      So the wheel package may fail on other GPU architectures.
      You can add -DCUDA_ARCH_NAME=All in cmake command
      to get a full wheel package to resolve this warning.
      While, this version will still work on local GPU architecture.")
    detect_installed_gpus(cuda_arch_bin)
  else() # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  if(NEW_RELEASE_JIT)
    set(cuda_arch_ptx "${cuda_arch_ptx}${cuda_arch_bin}")
    set(cuda_arch_bin "")
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+" cuda_arch_ptx "${cuda_arch_ptx}")

  list(REMOVE_DUPLICATES cuda_arch_bin)
  list(REMOVE_DUPLICATES cuda_arch_ptx)

  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      string(APPEND nvcc_flags
             " -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1}")
      string(APPEND nvcc_archs_readable " sm_${CMAKE_MATCH_1}")
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      string(APPEND nvcc_flags " -gencode arch=compute_${arch},code=sm_${arch}")
      string(APPEND nvcc_archs_readable " sm_${arch}")
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    string(APPEND nvcc_flags
           " -gencode arch=compute_${arch},code=compute_${arch}")
    string(APPEND nvcc_archs_readable " compute_${arch}")
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}
      ${nvcc_flags}
      PARENT_SCOPE)
  set(${out_variable}_readable
      ${nvcc_archs_readable}
      PARENT_SCOPE)
endfunction()

message(STATUS "CUDA detected: " ${CMAKE_CUDA_COMPILER_VERSION})
if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0) # CUDA 10.x
  set(fd_known_gpu_archs ${fd_known_gpu_archs10})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.2) # CUDA 11.0/11.1
  set(fd_known_gpu_archs ${fd_known_gpu_archs11})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0) # CUDA 11.2+
  set(fd_known_gpu_archs "${fd_known_gpu_archs11} 86")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
endif()

# setting nvcc arch flags
select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA}")
message(STATUS "NVCC_FLAGS_EXTRA: ${NVCC_FLAGS_EXTRA}")

# Set C++14 support
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
else()
  message(WARNING "Detected custom CMAKE_CUDA_STANDARD is using: ${CMAKE_CUDA_STANDARD}")  
endif()

# (Note) For windows, if delete /W[1-4], /W1 will be added defaultly and conflict with -w
# So replace /W[1-4] with /W0
if(WIN32)
  string(REGEX REPLACE "/W[1-4]" " /W0 " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()
# in cuda9, suppress cuda warning on eigen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
# Set :expt-relaxed-constexpr to suppress Eigen warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Set :expt-extended-lambda to enable HOSTDEVICE annotation on lambdas
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(WIN32)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd4244 /wd4267 /wd4819 \"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /bigobj")
  if(MSVC_STATIC_CRT)
    foreach(flag_var
            CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE
            CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "-MD")
        string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()
endif()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
