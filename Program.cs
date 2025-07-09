using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using Raylib_cs;
using static Raylib_cs.PixelFormat;
using System.Numerics;

using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace Madgett
{
    unsafe class PixelPlotterCUDA
    {
        enum FractalType { JULIA, MANDELBROT };

        // used as both input parameters and output results for the pixel Tasks
        class PixelTaskParams
        {
            public FractalType fractalType;
            public double minX, minY;
            public double xInc, yInc;
            public double cX, cY;
            public int maxIter;
            public int pixelX, pixelY;
            public int pixelWidth, pixelHeight;
            public Color* taskPixels;
            public double[,] taskIters;
            public int freeTaskPixelIdx;

            public PixelTaskParams(FractalType fractalType, double minX, double minY, double xInc, double yInc, double cX, double cY, int maxIter, int pixelX, int pixelY, int pixelWidth, int pixelHeight, Color* taskPixels, double[,] taskIters, int freeTaskPixelIdx)
            {
                this.fractalType = fractalType;
                this.minX = minX;
                this.minY = minY;
                this.xInc = xInc;
                this.yInc = yInc;
                this.cX = cX;
                this.cY = cY;
                this.maxIter = maxIter;
                this.pixelX = pixelX;
                this.pixelY = pixelY;
                this.pixelWidth = pixelWidth;
                this.pixelHeight = pixelHeight;
                this.taskPixels = taskPixels;
                this.taskIters = taskIters;
                this.freeTaskPixelIdx = freeTaskPixelIdx;
            }
        }

        //Color[] colours = (from c in Enumerable.Range(0, 256)
        //                   select new Color((c >> 5) * 36, (c >> 3 & 7) * 36, (c & 3) * 85, 255)).ToArray();
        //Color[] colours = (from c in Enumerable.Range(0, 256)
        //                   select new Color(c, c, c, 255)).ToArray();
        //Color[] colours = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW };

        Color[] colours;
        int numColours = 256;
        //Color[] colourPoints = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.RED };
        Color[] colourPoints = { Color.DARKBLUE, Color.GOLD, Color.MAROON, Color.ORANGE, Color.DARKBLUE };
        int[] colourIdx = { 0, 63, 127, 191, 255 };

        int screenPixelWidth = 1024;
        int screenPixelHeight = 1024;
        int taskPixelWidth = 64;
        int taskPixelHeight = 64;

        double[,] iterCount;

        public static void Main()
        {
            PixelPlotterCUDA plotter = new PixelPlotterCUDA();
            plotter.RunGame();
        }

        private Color[] GenerateColours()
        {
            Color[] genColours = new Color[numColours];
            if (colourPoints.Length != colourIdx.Length)
            {
                Console.WriteLine("ERROR: colourPoints.Length != colourIdx.Length");
            }
            int idxIdx = 0;
            int colourDist = 0;
            int idxDist = 0;
            Color prevColour = Color.WHITE;
            Color nextColour = Color.WHITE;
            for (int i = 0; i < numColours; i++)
            {
                if (i == colourIdx[idxIdx])
                {
                    genColours[i] = colourPoints[idxIdx];
                    prevColour = colourPoints[idxIdx];
                    if (idxIdx < colourIdx.Length - 1)
                    {
                        nextColour = colourPoints[idxIdx + 1];
                        colourDist = 0;
                        idxDist = colourIdx[idxIdx + 1] - colourIdx[idxIdx];
                        idxIdx++;
                    }
                }
                else
                {
                    float amount = colourDist / (float)idxDist;
                    Color newColour = LerpColor(prevColour, nextColour, amount);
                    genColours[i] = newColour;
                }

                colourDist++;
            }

            // duplicate the colours in reverse at the end of the (doubled) array
            Color[] doubledColours = new Color[genColours.Length * 2];
            for (int c = 0; c < genColours.Length; c++)
            {
                doubledColours[c] = genColours[c];
                doubledColours[(doubledColours.Length - 1) - c] = genColours[c];
            }
            genColours = doubledColours;

            return genColours;
        }

        public void RunGame()
        {
            colours = GenerateColours();

            //Raylib.SetConfigFlags(ConfigFlags.FLAG_VSYNC_HINT);
            Raylib.InitWindow(screenPixelWidth, screenPixelHeight, "PixelPlotter");
            //Raylib.SetTargetFPS(60);

            int currentMonitor = Raylib.GetCurrentMonitor();
            //screenPixelWidth = Raylib.GetMonitorWidth(currentMonitor);
            //screenPixelHeight = Raylib.GetMonitorHeight(currentMonitor);

            Raylib.SetWindowSize(screenPixelWidth, screenPixelHeight);
            //Raylib.SetWindowPosition(0, 0);
            //Raylib.ToggleFullscreen();

            FractalType fractalType = FractalType.MANDELBROT;

            double centreX = 0;
            double centreY = 0;
            double xSize = 3.5;
            double ySize = (xSize / screenPixelWidth) * screenPixelHeight;

            double minX = centreX - (xSize / 2);
            double maxX = centreX + (xSize / 2);
            double minY = centreY - (ySize / 2);
            double maxY = centreY + (ySize / 2);

            int maxIter = 1024;

            double xInc = (maxX - minX) / screenPixelWidth;
            double yInc = (maxY - minY) / screenPixelHeight;

            // pixels for the overall picture
            Color* screenPixels = (Color*)Raylib.MemAlloc(screenPixelWidth * screenPixelHeight * sizeof(Color));

            // create an Image of the right size using the pixels array as storage
            Image screenImage = new Image
            {
                data = screenPixels,
                width = screenPixelWidth,
                height = screenPixelHeight,
                format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
                mipmaps = 1,
            };

            // get the texture from the image
            Texture2D screenTexture = Raylib.LoadTextureFromImage(screenImage);

            // update the texture with the new pixels
            // TODO do we need this? not sure...
            Raylib.UpdateTexture(screenTexture, screenPixels);

            int N = screenPixelWidth * screenPixelHeight;
            int deviceID = 0;
            CudaContext ctx = new CudaContext(deviceID);
            CudaKernel mandelKernel = ctx.LoadKernel("Mandel.ptx", "Mandel");
            mandelKernel.GridDimensions = (N + 255) / 256;
            mandelKernel.BlockDimensions = 256;

            CudaKernel juliaKernel = ctx.LoadKernel("Julia.ptx", "Julia");
            juliaKernel.GridDimensions = (N + 255) / 256;
            juliaKernel.BlockDimensions = 256;

            // Allocate vectors in device memory and copy vectors from host memory to device memory 
            CudaDeviceVariable<int> d_counts = new CudaDeviceVariable<int>(N);

            // Invoke kernel
            float time = mandelKernel.Run(screenPixelWidth, screenPixelHeight, (float)minX, (float)maxX, (float)minY, (float)maxY, (float)xInc, (float)yInc, maxIter, d_counts.DevicePointer, N);
            Console.WriteLine(time);  

            // Copy result from device memory to host memory
            // h_C contains the result in host memory
            int[] h_counts = d_counts;

            for(int y = 0; y < screenPixelHeight; y++)
            {
                for(int x = 0; x < screenPixelWidth; x++)
                {
                    int iter = h_counts[(y * screenPixelWidth) + x];
                    Color col = GetColourFromIter(iter, maxIter);
                    SetColor(screenPixels, x, y, col);
                }
            }

            Raylib.UpdateTexture(screenTexture, screenPixels);

            bool drawMandel = false;
            bool juliaCompleted = true;
            long lastParamChangeTime = GetCurrentMilliseconds();
            double juliaStartCentreX = 0;
            double juliaStartCentreY = 0;
            double juliaXSize = 0;
            double juliaYSize = 0;
            double juliaStartMinX = 0;
            double juliaStartMaxX = 0;
            double juliaStartMinY = 0;
            double juliaStartMaxY = 0;
            double juliaXInc = 0;
            double juliaYInc = 0;
            double juliaEndCentreX = 0;
            double juliaEndCentreY = 0;
            double juliaEndMinX = 0;
            double juliaEndMaxX = 0;
            double juliaEndMinY = 0;
            double juliaEndMaxY = 0;
            int juliaCurrentFrame = 0;
            int juliaNumFrames = 512;
            double juliaFrameXInc = 0;
            double juliaFrameYInc = 0;

            while (!Raylib.WindowShouldClose())
            {
                Raylib.BeginDrawing();

                Raylib.ClearBackground(Color.WHITE);

                // draw the main texture to the screen
                Raylib.DrawTexture(screenTexture, 0, 0, Color.WHITE);

                // have less than 3s passed since last change of maxIter?
                long currentTime = GetCurrentMilliseconds();
                if (currentTime - lastParamChangeTime < 3000)
                {
                    // display current maxIter if so
                    Raylib.DrawText("(centreX, centreY): (" + centreX + ", " + centreY + ")", 16, 16, 24, Color.WHITE);
                    Raylib.DrawText("xSize: " + xSize + " ySize: " + ySize, 16, 64, 24, Color.WHITE);
                    Raylib.DrawText("maxIter: " + maxIter, 16, 112, 24, Color.WHITE);
                    Raylib.DrawText("frameTime: " + time, 16, 164, 24, Color.WHITE);
                }

                Raylib.EndDrawing();

                if (drawMandel)
                {
                    // Invoke kernel
                    time = mandelKernel.Run(screenPixelWidth, screenPixelHeight, (float)minX, (float)maxX, (float)minY, (float)maxY, (float)xInc, (float)yInc, maxIter, d_counts.DevicePointer, N);
                    Console.WriteLine(time);

                    // Copy result from device memory to host memory
                    // h_C contains the result in host memory
                    h_counts = d_counts;

                    for (int y = 0; y < screenPixelHeight; y++)
                    {
                        for (int x = 0; x < screenPixelWidth; x++)
                        {
                            int iter = h_counts[(y * screenPixelWidth) + x];
                            Color col = GetColourFromIter(iter, maxIter);
                            SetColor(screenPixels, x, y, col);
                        }
                    }

                    Raylib.UpdateTexture(screenTexture, screenPixels);
                    drawMandel = false;
                }

                if (!juliaCompleted)
                {
                    double juliaCurrentMinX = juliaStartMinX + (juliaFrameXInc * juliaCurrentFrame);
                    double juliaCurrentMaxX = juliaStartMaxX + (juliaFrameXInc * juliaCurrentFrame);
                    double juliaCurrentMinY = juliaStartMinY + (juliaFrameYInc * juliaCurrentFrame);
                    double juliaCurrentMaxY = juliaStartMaxY + (juliaFrameYInc * juliaCurrentFrame);
                    double juliaCurrentCentreX = juliaStartCentreX + (juliaFrameXInc * juliaCurrentFrame);
                    double juliaCurrentCentreY = juliaStartCentreY + (juliaFrameYInc * juliaCurrentFrame);

                    // Invoke kernel
                    time = juliaKernel.Run(screenPixelWidth, screenPixelHeight, (float)juliaCurrentMinX, (float)juliaCurrentMaxX, (float)juliaCurrentMinY, (float)juliaCurrentMinY, (float)juliaXInc, (float)juliaYInc, (float)juliaCurrentCentreX, (float)juliaCurrentCentreY, maxIter, d_counts.DevicePointer, N);
                    Console.WriteLine(time);

                    // Copy result from device memory to host memory
                    // h_C contains the result in host memory
                    h_counts = d_counts;

                    for (int y = 0; y < screenPixelHeight; y++)
                    {
                        for (int x = 0; x < screenPixelWidth; x++)
                        {
                            int iter = h_counts[(y * screenPixelWidth) + x];
                            Color col = GetColourFromIter(iter, maxIter);
                            SetColor(screenPixels, x, y, col);
                        }
                    }

                    Raylib.UpdateTexture(screenTexture, screenPixels);

                    juliaCurrentFrame++;
                    if(juliaCurrentFrame == juliaNumFrames)
                        juliaCompleted = true;
                }

                //get start pos
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_S))
                {
                    Vector2 startPos = Raylib.GetMousePosition();
                    juliaStartCentreX = minX + (startPos.X * xInc);
                    juliaStartCentreY = maxY - (startPos.Y * yInc);
                    juliaXSize = xSize;
                    juliaYSize = (juliaXSize / screenPixelWidth) * screenPixelHeight;

                    juliaStartMinX = juliaStartCentreX - (juliaXSize / 2);
                    juliaStartMaxX = juliaStartCentreX + (juliaXSize / 2);
                    juliaStartMinY = juliaStartCentreY - (juliaYSize / 2);
                    juliaStartMaxY = juliaStartCentreY + (juliaYSize / 2);

                    juliaXInc = (juliaStartMaxX - juliaStartMinX) / screenPixelWidth;
                    juliaYInc = (juliaStartMaxY - juliaStartMinY) / screenPixelHeight;
                }

                //draw mandel
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_M))
                {
                    drawMandel = true;
                }

                //get end pos
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_E))
                {
                    Vector2 endPos = Raylib.GetMousePosition();
                    juliaEndCentreX = minX + (endPos.X * xInc);
                    juliaEndCentreY = maxY - (endPos.Y * yInc);

                    juliaEndMinX = juliaEndCentreX - (juliaXSize / 2);
                    juliaEndMaxX = juliaEndCentreX + (juliaXSize / 2);
                    juliaEndMinY = juliaEndCentreY - (juliaYSize / 2);
                    juliaEndMaxY = juliaEndCentreY + (juliaYSize / 2);
                    juliaCurrentFrame = 0;
                    juliaFrameXInc = (juliaEndCentreX - juliaStartCentreX) / juliaNumFrames;
                    juliaFrameYInc = (juliaEndCentreY - juliaStartCentreY) / juliaNumFrames;
                }

                if (Raylib.IsKeyPressed(KeyboardKey.KEY_J))
                {
                    juliaCompleted = false;
                }

                // zoom in 2x
                if (Raylib.IsMouseButtonPressed(MouseButton.MOUSE_LEFT_BUTTON))
                {
                    Vector2 mousePos = Raylib.GetMousePosition();

                    centreX = minX + (mousePos.X * xInc);
                    centreY = maxY - (mousePos.Y * yInc);
                    xSize = xSize / 2;
                    ySize = (xSize / screenPixelWidth) * screenPixelHeight;

                    minX = centreX - (xSize / 2);
                    maxX = centreX + (xSize / 2);
                    minY = centreY - (ySize / 2);
                    maxY = centreY + (ySize / 2);

                    xInc = (maxX - minX) / screenPixelWidth;
                    yInc = (maxY - minY) / screenPixelHeight;

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                // zoom out 2x
                if (Raylib.IsMouseButtonPressed(MouseButton.MOUSE_RIGHT_BUTTON))
                {
                    Vector2 mousePos = Raylib.GetMousePosition();

                    centreX = minX + (mousePos.X * xInc);
                    centreY = maxY - (mousePos.Y * yInc);
                    xSize = xSize * 2;
                    ySize = (xSize / screenPixelWidth) * screenPixelHeight;

                    minX = centreX - (xSize / 2);
                    maxX = centreX + (xSize / 2);
                    minY = centreY - (ySize / 2);
                    maxY = centreY + (ySize / 2);

                    xInc = (maxX - minX) / screenPixelWidth;
                    yInc = (maxY - minY) / screenPixelHeight;

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                // zoom in a little
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_KP_ADD))
                {
                    Vector2 mousePos = Raylib.GetMousePosition();

                    centreX = minX + (mousePos.X * xInc);
                    centreY = maxY - (mousePos.Y * yInc);
                    xSize = xSize * 0.9;
                    ySize = (xSize / screenPixelWidth) * screenPixelHeight;

                    minX = centreX - (xSize / 2);
                    maxX = centreX + (xSize / 2);
                    minY = centreY - (ySize / 2);
                    maxY = centreY + (ySize / 2);

                    xInc = (maxX - minX) / screenPixelWidth;
                    yInc = (maxY - minY) / screenPixelHeight;

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                // zoom out a little
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_KP_SUBTRACT))
                {
                    Vector2 mousePos = Raylib.GetMousePosition();

                    centreX = minX + (mousePos.X * xInc);
                    centreY = maxY - (mousePos.Y * yInc);
                    xSize = xSize / 0.9;
                    ySize = (xSize / screenPixelWidth) * screenPixelHeight;

                    minX = centreX - (xSize / 2);
                    maxX = centreX + (xSize / 2);
                    minY = centreY - (ySize / 2);
                    maxY = centreY + (ySize / 2);

                    xInc = (maxX - minX) / screenPixelWidth;
                    yInc = (maxY - minY) / screenPixelHeight;

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                if (Raylib.IsKeyPressed(KeyboardKey.KEY_B))
                {
                    juliaCompleted = true;
                    drawMandel = true;
                }

                // recentre
                if (Raylib.IsMouseButtonPressed(MouseButton.MOUSE_MIDDLE_BUTTON))
                {
                    Vector2 mousePos = Raylib.GetMousePosition();

                    centreX = minX + (mousePos.X * xInc);
                    centreY = maxY - (mousePos.Y * yInc);

                    minX = centreX - (xSize / 2);
                    maxX = centreX + (xSize / 2);
                    minY = centreY - (ySize / 2);
                    maxY = centreY + (ySize / 2);

                    xInc = (maxX - minX) / screenPixelWidth;
                    yInc = (maxY - minY) / screenPixelHeight;

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                // show parameters for a few seconds
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_P))
                {
                    lastParamChangeTime = GetCurrentMilliseconds();
                }

                // take a screenshot
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_SPACE))
                {
                    Image currentScreen = Raylib.LoadImageFromScreen();
                    string currentDateTime = DateTime.Now.ToString("yyyyMMddHHmmss");
                    string coords = centreX + "_" + centreY + "_" + xSize + "_" + maxIter + "_";
                    Raylib.ExportImage(currentScreen, "PixelPlotter_" + coords + currentDateTime + ".png");
                }

                // Double maxIter
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_KP_MULTIPLY))
                {
                    maxIter *= 2;
                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                // Halve maxIter
                if (Raylib.IsKeyPressed(KeyboardKey.KEY_KP_DIVIDE))
                {
                    maxIter /= 2;
                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }

                if (Raylib.IsKeyPressed(KeyboardKey.KEY_F))
                {
                    if (fractalType == FractalType.JULIA)
                    {
                        fractalType = FractalType.MANDELBROT;
                    }
                    else
                    {
                        fractalType = FractalType.JULIA;
                    }

                    lastParamChangeTime = GetCurrentMilliseconds();
                    drawMandel = true;
                }
            }

            Raylib.MemFree(screenPixels);
            Raylib.UnloadTexture(screenTexture);  

            Raylib.CloseWindow();
        }

        private long GetCurrentMilliseconds()
        {
            return System.DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
        }

        // set a colour in the main pixels
        private void SetColor(Color* screenPixels, int x, int y, Color color)
        {
            int index = x + (y * screenPixelWidth);

            screenPixels[index] = color;
        }

        private Color GetColourFromIter(int iter, int maxIter)
        {
            Color col;
            if (iter == maxIter)
            {
                col = Color.BLACK;
            }
            else
            {
                // look up colour based on iteration count
                //i = i % colours.Length-1; // modulus on doubles is weird! do it ourselves
                double div = Math.Truncate((double)iter / (double)colours.Length);
                double subNo = colours.Length * div;
                double i = iter - subNo;

                int thisI = (int)i;
                Color col1 = colours[thisI];
                int nextI = thisI + 1;
                if (nextI >= colours.Length) nextI = nextI - colours.Length;
                Color col2 = colours[nextI];
                float amount = (float)(i - Math.Truncate(i));
                col = LerpColor(col1, col2, amount);
            }
            return col;
        }

        private Color LerpColor (Color col1, Color col2, float amount)
        {
            int r = col1.r + (int)((col2.r - col1.r) * amount);
            int g = col1.g + (int)((col2.g - col1.g) * amount);
            int b = col1.b + (int)((col2.b - col1.b) * amount);
            int a = col1.a + (int)((col2.a - col1.a) * amount);

            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;
            if (a > 255) a = 255;

            Color col = new Color(r, g, b, a);
            return col;
        }
    }
}