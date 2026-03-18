cmake_minimum_required(VERSION 3.22)
project(DeFeedbackPro VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS: build universal binary (Intel + Apple Silicon)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build universal binary")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

# JUCE — fetched from GitHub
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 7.0.12
)
FetchContent_MakeAvailable(JUCE)

# PFFFT — vendored source
add_library(pffft STATIC third_party/pffft/pffft.c)
target_include_directories(pffft PUBLIC third_party/pffft)
if(MSVC)
    target_compile_options(pffft PRIVATE /arch:SSE2)
elseif(APPLE)
    target_compile_definitions(pffft PRIVATE $<$<STREQUAL:${CMAKE_OSX_ARCHITECTURES},arm64>:PFFFT_ENABLE_NEON>)
else()
    target_compile_options(pffft PRIVATE -msse2)
endif()

# ONNX Runtime — prebuilt, set ONNXRUNTIME_ROOT via cmake -D
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib)

# Plugin target
juce_add_plugin(DeFeedbackPro
    COMPANY_NAME "DeFeedbackPro"
    PLUGIN_MANUFACTURER_CODE DfPr
    PLUGIN_CODE DfPr
    FORMATS VST3 AU
    PRODUCT_NAME "DeFeedback Pro"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    BUNDLE_ID "com.defeedbackpro.plugin"
    AU_MAIN_TYPE "kAudioUnitType_Effect"
    COPY_PLUGIN_AFTER_BUILD FALSE
)

target_sources(DeFeedbackPro PRIVATE
    modules/PluginProcessor.cpp
    modules/PluginEditor.cpp
    modules/SpectralEngine.cpp
    modules/FeedbackDetector.cpp
    modules/SpectralSubtractor.cpp
    modules/MaskCombiner.cpp
    modules/MLProcessor.cpp
    modules/SpectrumDisplay.cpp
)

target_include_directories(DeFeedbackPro PRIVATE
    modules
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(DeFeedbackPro PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    pffft
    ${ONNXRUNTIME_LIB}
)

juce_generate_juce_header(DeFeedbackPro)

target_compile_definitions(DeFeedbackPro PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)
target_compile_definitions(DeFeedbackPro PRIVATE _USE_MATH_DEFINES)

# Copy models to build output
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/models/*.onnx")
foreach(MODEL ${MODEL_FILES})
    get_filename_component(MODEL_NAME ${MODEL} NAME)
    configure_file(${MODEL} ${CMAKE_BINARY_DIR}/models/${MODEL_NAME} COPYONLY)
endforeach()

# macOS: embed models into plugin bundle Resources
if(APPLE)
    foreach(MODEL ${MODEL_FILES})
        get_filename_component(MODEL_NAME ${MODEL} NAME)
        target_sources(DeFeedbackPro PRIVATE ${MODEL})
        set_source_files_properties(${MODEL} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources/models")
    endforeach()

    add_custom_command(TARGET DeFeedbackPro POST_BUILD
        COMMAND codesign --force --sign "-" --deep
            "$<TARGET_BUNDLE_DIR:DeFeedbackPro>"
        COMMENT "Ad-hoc codesigning plugin bundle"
    )
endif()

# Tests (optional, via Catch2)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(DeFeedbackProTests
        tests/RingBufferTest.cpp
        tests/SpectralEngineTest.cpp
        tests/FeedbackDetectorTest.cpp
        tests/SpectralSubtractorTest.cpp
        tests/MaskCombinerTest.cpp
        tests/MLProcessorTest.cpp
        modules/SpectralEngine.cpp
        modules/FeedbackDetector.cpp
        modules/SpectralSubtractor.cpp
        modules/MaskCombiner.cpp
        modules/MLProcessor.cpp
    )
    target_include_directories(DeFeedbackProTests PRIVATE
        modules
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(DeFeedbackProTests PRIVATE
        Catch2::Catch2WithMain
        pffft
        ${ONNXRUNTIME_LIB}
    )
    target_compile_definitions(DeFeedbackProTests PRIVATE
        DEFEEDBACK_TEST_MODE=1
        _USE_MATH_DEFINES
    )
endif()
