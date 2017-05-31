add_library(ddPCM-tests OBJECT ddPCM.cpp)
if(BUILD_CUSTOM_BOOST)
  add_dependencies(ddPCM-tests custom_boost)
endif()

# Copy reference files to ${PROJECT_BINARY_DIR}/tests/ddPCM (aka ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND _files grid.npy ggrid.npy xi.npy phi.npy)
file(COPY ${_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ddPCM.cpp test
add_Catch_test(ddPCM "ddPCM;ddCOSMO")

