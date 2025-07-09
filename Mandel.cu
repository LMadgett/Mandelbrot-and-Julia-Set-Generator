extern "C"
{
	//Device code
	__global__ void Mandel(const int pixWidth, const int pixHeight, const float minX, const float maxX,
	const float minY, const float maxY, const float xInc, const float yInc, const int maxIter, int* counts, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < N)
		{
			float zReal = 0.0;
			float zImg = 0.0;
			int xPos = i % pixWidth;
			int yPos = i / pixWidth;
			float cReal = minX + (xPos * xInc);
			float cImg = maxY - (yPos * yInc);
			int iter = 0;
			while((zReal * zReal) + (zImg * zImg) <= 4.0 && (iter < maxIter))
			{
				float nextZReal = (zReal * zReal) - (zImg * zImg) + cReal;
				float nextZImg = (2.0 * zReal * zImg) + cImg;
				zReal = nextZReal;
				zImg = nextZImg;
				iter = iter + 1;
			}
			counts[i] = iter;
		}
	}
}