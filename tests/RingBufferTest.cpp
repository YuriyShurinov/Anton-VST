#include <catch2/catch_all.hpp>
#include "RingBuffer.h"

TEST_CASE("RingBuffer: push and pop single element", "[ringbuffer]")
{
    RingBuffer<float> rb(16);
    float val = 42.0f;
    REQUIRE(rb.push(&val, 1));
    float out = 0.0f;
    REQUIRE(rb.pop(&out, 1));
    REQUIRE(out == 42.0f);
}

TEST_CASE("RingBuffer: pop from empty returns false", "[ringbuffer]")
{
    RingBuffer<float> rb(16);
    float out = 0.0f;
    REQUIRE_FALSE(rb.pop(&out, 1));
}

TEST_CASE("RingBuffer: push to full returns false", "[ringbuffer]")
{
    RingBuffer<float> rb(4);
    float data[4] = {1, 2, 3, 4};
    REQUIRE(rb.push(data, 4));
    float extra = 5.0f;
    REQUIRE_FALSE(rb.push(&extra, 1));
}

TEST_CASE("RingBuffer: wrap-around works correctly", "[ringbuffer]")
{
    RingBuffer<float> rb(4);
    float data[3] = {1, 2, 3};
    REQUIRE(rb.push(data, 3));
    float out[3];
    REQUIRE(rb.pop(out, 3));

    float data2[3] = {4, 5, 6};
    REQUIRE(rb.push(data2, 3));
    float out2[3];
    REQUIRE(rb.pop(out2, 3));
    REQUIRE(out2[0] == 4.0f);
    REQUIRE(out2[1] == 5.0f);
    REQUIRE(out2[2] == 6.0f);
}

TEST_CASE("RingBuffer: availableToRead reports correctly", "[ringbuffer]")
{
    RingBuffer<float> rb(8);
    REQUIRE(rb.availableToRead() == 0);
    float data[3] = {1, 2, 3};
    rb.push(data, 3);
    REQUIRE(rb.availableToRead() == 3);
}
