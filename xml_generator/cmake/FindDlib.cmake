# - Find Dlib
# Find the native Dlib includes and library
#
#  DLIB_INCLUDE_DIR - where to find dlib headers, etc.
#  DLIB_LIBRARIES   - List of libraries when using dlib.
#  DLIB_FOUND       - True if dlib is found.

include(FindPackageHandleStandardArgs)

find_package(X11 REQUIRED)
find_package(BLAS REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(LAPACK REQUIRED)
# on some systems it is installed (or gettting installed) and if present then
# dlib uses it and hence we need to include its library as well in the linking
# part. 
find_package(GIF)   # this is an optional package and will set GIF_LIBRARY

# Look for headers
find_path(DLIB_INCLUDE_DIR NAMES dlib/algs.h PATHS $ENV{DLIB_INCLUDE} /opt/local/include /usr/local/include /usr/include DOC "Path in which the file dlib is located." )
mark_as_advanced(DLIB_INCLUDE_DIR)

find_library(DLIB_LIBRARIES NAMES dlib libdlib PATHS /usr/lib /usr/local/lib DOC "Path to dlib library." )
mark_as_advanced(DLIB_LIBRARIES)

if (DLIB_INCLUDE_DIR AND DLIB_LIBRARIES)  
  set(DLIB_FOUND 1)  
else ()
   set(DLIB_FOUND 0)
endif ()

# Report the results.
if (NOT DLIB_FOUND)
    set(DLIB_DIR_MESSAGE "Dlib was not found")
    message(FATAL_ERROR "${DLIB_DIR_MESSAGE}")   
endif ()

if (GIF_FOUND)
set (DLIB_LIBRARIES ${DLIB_LIBRARIES} ${GIF_LIBRARY})
endif()

if(DLIB_FOUND AND NOT TARGET dlib::dlib)
    add_library(dlib::dlib INTERFACE IMPORTED)
    set_property(TARGET dlib::dlib PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DLIB_INCLUDE_DIR})
    set_property(TARGET dlib::dlib PROPERTY INTERFACE_LINK_LIBRARIES ${DLIB_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${X11_LIBRARIES})    
endif()
