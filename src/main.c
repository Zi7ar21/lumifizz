#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "noise1234.h"

// https://nullprogram.com/blog/2018/07/31/ exact bias: 0.020888578919738908
inline uint32_t triple32(uint32_t x) {
	x ^= x >> (uint32_t)17u;
	x *= (uint32_t)0xED5AD4BBu;
	x ^= x >> (uint32_t)11u;
	x *= (uint32_t)0xAC4C1B51u;
	x ^= x >> (uint32_t)15u;
	x *= (uint32_t)0x31848BABu;
	x ^= x >> (uint32_t)14u;
	return x;
}

int main(void) {
	int image_size_x = 1920, image_size_y = 1080, tile_size_x = 10, tile_size_y = 10;

	int num_tiles_x = image_size_x / tile_size_x, num_tiles_y = image_size_y / tile_size_y;

	printf("Selected Parameters:\n{\n  Image Size: %dx%d\n  Tile Size: %dx%d\n}\n", image_size_x, image_size_y, tile_size_x, tile_size_y);

	// verify parameters are valid (sanity check)
	if(image_size_x < 1 || image_size_y < 1 || tile_size_x < 1 || tile_size_y < 1 || tile_size_x > image_size_x || tile_size_y > image_size_y) {
		if(image_size_x < 1) printf("Error: image_size_x (%d) must be at least 1!\n", image_size_x);
		if(image_size_y < 1) printf("Error: image_size_y (%d) must be at least 1!\n", image_size_y);
		if(tile_size_x < 1) printf("Error: tile_size_x (%d) must be at least 1!\n", tile_size_x);
		if(tile_size_y < 1) printf("Error: tile_size_y (%d) must be at least 1!\n", tile_size_y);
		if(tile_size_x > image_size_x) printf("Error: tile_size_x (%d) cannot be larger than image_size_x (%d)!\n", tile_size_x, image_size_x);
		if(tile_size_y > image_size_y) printf("Error: tile_size_y (%d) cannot be larger than image_size_y (%d)!\n", tile_size_y, image_size_y);

		return EXIT_FAILURE;
	}

	// while the image can still be rendered with a non-whole number of tiles, there will be black borders (since rendering is skipped to prevent writing out-of-bounds)
	if(image_size_x % tile_size_x != 0) printf("Warning: tile_size_x (%d) is not a multiple of image_size_x (%d)!\n", tile_size_x, image_size_x);
	if(image_size_y % tile_size_y != 0) printf("Warning: tile_size_y (%d) is not a multiple of image_size_y (%d)!\n", tile_size_y, image_size_y);

	fflush(stdout);

	size_t image_size = image_size_x * image_size_y * 3 * sizeof(float); // RGB32F

	printf("\nAllocating a %zu B image buffer...", image_size);

	fflush(stdout);

	float* image = (float*)malloc(image_size); // allocate image buffer

	if(image == NULL) {
		perror("\nmalloc() returned NULL! Error");

		return EXIT_FAILURE;
	}

	printf(" Done!\n\nInitializing...");

	fflush(stdout);

	{
		#pragma omp simd
		for(int i = 0; i < image_size_x * image_size_y * 3; i++) image[i] = 0.0f; // initialize image buffer
	}

	printf(" Done!\n\nStarting render...");

	fflush(stdout);

	#pragma omp parallel for schedule(dynamic) collapse(2)
	for(int tile_y = num_tiles_y - 1; tile_y >= 0; tile_y--) {
		for(int tile_x = 0; tile_x < num_tiles_x; tile_x++) {
			#pragma omp simd collapse(2)
			for(int local_y = tile_size_y - 1; local_y >= 0; local_y--) {
				for(int local_x = 0; local_x < tile_size_x; local_x++) {
					int coord_x = tile_size_x * tile_x + local_x, coord_y = tile_size_y * tile_y + local_y;

					const int spp = 16;

					for(int n = 0; n < spp; n++) {
					uint32_t ns =
					((uint32_t)n * (uint32_t)image_size_x * (uint32_t)image_size_y) +
					((uint32_t)image_size_x * (uint32_t)coord_y + (uint32_t)coord_x) +
					(uint32_t)1u;

					ns = triple32(ns);
					float pixel_coord_x = (float)coord_x + (0.00000000023283064365386962890625f * (float)ns);
					ns = triple32(ns);
					float pixel_coord_y = (float)coord_y + (0.00000000023283064365386962890625f * (float)ns);
					ns = triple32(ns);
					float dither = 0.00000000023283064365386962890625f * (float)ns;

					// screen-space uv coordinates
					float uv[2] = {
					(pixel_coord_x - 0.5f * (float)image_size_x) / (float)image_size_y,
					(pixel_coord_y - 0.5f * (float)image_size_y) / (float)image_size_y};

					float ro[3] = { 0.000f,  0.000f,  5.000f};
					float rd[3] = {  uv[0],   uv[1], -1.000f};

					// normalize ray direction
					{
						float _r = sqrtf(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]);

						rd[0] /= _r;
						rd[1] /= _r;
						rd[2] /= _r;
					}

					float rad[3] = {0.0f, 0.0f, 0.0f}; // radiance

					for(int i = 0; i < 64; i++) {
						const float step_size = 0.15625f;

						float t = step_size * ((float)i + dither);

						float x[3] = {
						t * rd[0] + ro[0],
						t * rd[1] + ro[1],
						t * rd[2] + ro[2]};

						const float nscale = 1.0f;

						float pnoise = 0.0f;

						if(x[0]*x[0]+x[1]*x[1]+x[2]*x[2] < 2.0f * 2.0f) {
							pnoise =
							0.5000f * noise3(nscale * 1.0f * x[0], nscale * 1.0f * x[1], nscale * 1.0f * x[2]) +
							0.2500f * noise3(nscale * 2.0f * x[0], nscale * 2.0f * x[1], nscale * 2.0f * x[2]) +
							0.1250f * noise3(nscale * 4.0f * x[0], nscale * 4.0f * x[1], nscale * 4.0f * x[2]) +
							0.0625f * noise3(nscale * 8.0f * x[0], nscale * 8.0f * x[1], nscale * 8.0f * x[2]);

							pnoise = fmaxf(1.0f - ((16.0f * pnoise) * (16.0 * pnoise)), 0.0);
						}

						float density = expf(-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])) * pnoise;

						rad[0] += step_size * density;
						rad[1] += step_size * density;
						rad[2] += step_size * density;
					}

					image[3 * (image_size_x * coord_y + coord_x) + 0] += 2.0f * rad[0];
					image[3 * (image_size_x * coord_y + coord_x) + 1] += 2.0f * rad[1];
					image[3 * (image_size_x * coord_y + coord_x) + 2] += 2.0f * rad[2];
					}

					image[3 * (image_size_x * coord_y + coord_x) + 0] /= (float)spp;
					image[3 * (image_size_x * coord_y + coord_x) + 1] /= (float)spp;
					image[3 * (image_size_x * coord_y + coord_x) + 2] /= (float)spp;
				}
			}
		}
	}

	const char filename[] = "./image.hdr";

	printf(" Done!\n\nWriting rendered image to \"%s\"...", filename);

	fflush(stdout);

	if(stbi_write_hdr(filename, image_size_x, image_size_y, 3, image) != 0) {
		printf(" Done!");
	} else {
		printf("\nError: stbi_write_hdr() failed!");
	}

	printf("\n\nFreeing the %zu B image buffer...", image_size);

	fflush(stdout);

	free(image);

	printf(" Done!\n\n");

	printf("Finished!\n");

	return EXIT_SUCCESS;
}
