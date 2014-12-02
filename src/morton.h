

/*
 *
 * https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
 *
 */

#ifndef _MORTON_H
#define _MORTON_H

#include <vector>
#include "cudaQualifiers.h"

HOST void calcMorton(int width, int height, std::vector<unsigned int>& array);
HOST void getCudaMortonBuffer(std::vector<unsigned int>& array, unsigned int** buffer, int* buffer_len);

// "Insert" a 0 bit after each of the 16 low bits of x
HOST DEVICE inline unsigned int Part1By1(unsigned int x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
HOST DEVICE inline unsigned int Part1By2(unsigned int x)
{
  x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}

HOST DEVICE inline unsigned int EncodeMorton2(unsigned int x, unsigned int y)
{
  return (Part1By1(y) << 1) + Part1By1(x);
}

HOST DEVICE inline unsigned int EncodeMorton3(unsigned int x, unsigned int y, unsigned int z)
{
  return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}



// Inverse of Part1By1 - "delete" all odd-indexed bits
HOST DEVICE inline unsigned int Compact1By1(unsigned int x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
HOST DEVICE inline unsigned int Compact1By2(unsigned int x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}


HOST DEVICE inline unsigned int DecodeMorton2X(unsigned int code)
{
  return Compact1By1(code >> 0);
}

HOST DEVICE inline unsigned int DecodeMorton2Y(unsigned int code)
{
  return Compact1By1(code >> 1);
}

HOST DEVICE inline unsigned int DecodeMorton3X(unsigned int code)
{
  return Compact1By2(code >> 0);
}

HOST DEVICE inline unsigned int DecodeMorton3Y(unsigned int code)
{
  return Compact1By2(code >> 1);
}

HOST DEVICE inline unsigned int DecodeMorton3Z(unsigned int code)
{
  return Compact1By2(code >> 2);
}



#endif