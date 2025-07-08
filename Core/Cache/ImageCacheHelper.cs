
// ===== Core/Cache/ImageCacheHelper.cs =====
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace WabiSabiRevitBridge.Core.Cache
{
    /// <summary>
    /// Specialized cache helper for image data with perceptual hashing
    /// </summary>
    public static class ImageCacheHelper
    {
        /// <summary>
        /// Generates a perceptual hash for an image to detect similar images
        /// </summary>
        public static string ComputePerceptualHash(byte[] imageData)
        {
            try
            {
                using (var ms = new MemoryStream(imageData))
                using (var bitmap = new Bitmap(ms))
                {
                    // Resize to 8x8
                    using (var small = new Bitmap(8, 8))
                    using (var graphics = Graphics.FromImage(small))
                    {
                        graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                        graphics.DrawImage(bitmap, 0, 0, 8, 8);
                        
                        // Convert to grayscale and compute average
                        var pixels = new byte[64];
                        var sum = 0;
                        
                        for (int y = 0; y < 8; y++)
                        {
                            for (int x = 0; x < 8; x++)
                            {
                                var pixel = small.GetPixel(x, y);
                                var gray = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                                pixels[y * 8 + x] = gray;
                                sum += gray;
                            }
                        }
                        
                        var average = sum / 64;
                        
                        // Generate hash
                        var hash = 0UL;
                        for (int i = 0; i < 64; i++)
                        {
                            if (pixels[i] > average)
                            {
                                hash |= (1UL << i);
                            }
                        }
                        
                        return hash.ToString("X16");
                    }
                }
            }
            catch
            {
                // Fallback to regular hash
                return ComputeHash(imageData);
            }
        }

        /// <summary>
        /// Computes Hamming distance between two perceptual hashes
        /// </summary>
        public static int ComputeHammingDistance(string hash1, string hash2)
        {
            if (hash1.Length != hash2.Length)
                return int.MaxValue;
            
            try
            {
                var h1 = Convert.ToUInt64(hash1, 16);
                var h2 = Convert.ToUInt64(hash2, 16);
                
                var xor = h1 ^ h2;
                var distance = 0;
                
                while (xor != 0)
                {
                    distance++;
                    xor &= xor - 1;
                }
                
                return distance;
            }
            catch
            {
                return int.MaxValue;
            }
        }

        /// <summary>
        /// Checks if two images are similar based on perceptual hash
        /// </summary>
        public static bool AreSimilar(string hash1, string hash2, int threshold = 5)
        {
            var distance = ComputeHammingDistance(hash1, hash2);
            return distance <= threshold;
        }

        /// <summary>
        /// Computes standard SHA256 hash for exact matching
        /// </summary>
        public static string ComputeHash(byte[] data)
        {
            using (var sha256 = SHA256.Create())
            {
                var hash = sha256.ComputeHash(data);
                return BitConverter.ToString(hash).Replace("-", "").ToLower();
            }
        }

        /// <summary>
        /// Generates a cache key for image data including metadata
        /// </summary>
        public static string GenerateImageCacheKey(string viewName, int width, int height, 
            string exportType, string additionalParams = null)
        {
            var keyBuilder = new StringBuilder();
            keyBuilder.Append($"img_{viewName}_{width}x{height}_{exportType}");
            
            if (!string.IsNullOrEmpty(additionalParams))
            {
                keyBuilder.Append($"_{additionalParams}");
            }
            
            return ComputeHash(Encoding.UTF8.GetBytes(keyBuilder.ToString()));
        }
    }
}