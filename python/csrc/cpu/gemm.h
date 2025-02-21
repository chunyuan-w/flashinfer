#pragma once

#include <ATen/native/CPUBlas.h>

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 2
#define TILE_SIZE 512
