
// ===== Utils/ImageHelper.cs =====
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

namespace WabiSabiRevitBridge.Utils
{
    /// <summary>
    /// Helper methods for image processing
    /// </summary>
    public static class ImageHelper
    {
        public static byte[] ResizeImage(byte[] imageData, int targetWidth, int targetHeight, 
            InterpolationMode interpolation = InterpolationMode.HighQualityBicubic)
        {
            using (var ms = new MemoryStream(imageData))
            using (var originalImage = Image.FromStream(ms))
            {
                using (var resizedBitmap = new Bitmap(targetWidth, targetHeight))
                using (var graphics = Graphics.FromImage(resizedBitmap))
                {
                    graphics.InterpolationMode = interpolation;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    
                    graphics.DrawImage(originalImage, 0, 0, targetWidth, targetHeight);
                    
                    using (var output = new MemoryStream())
                    {
                        resizedBitmap.Save(output, ImageFormat.Png);
                        return output.ToArray();
                    }
                }
            }
        }

        public static byte[] ConvertToFormat(byte[] imageData, ImageFormat targetFormat, int quality = 90)
        {
            using (var ms = new MemoryStream(imageData))
            using (var image = Image.FromStream(ms))
            {
                using (var output = new MemoryStream())
                {
                    if (targetFormat.Equals(ImageFormat.Jpeg))
                    {
                        var encoderParameters = new EncoderParameters(1);
                        encoderParameters.Param[0] = new EncoderParameter(Encoder.Quality, quality);
                        
                        var jpegCodec = GetEncoder(ImageFormat.Jpeg);
                        image.Save(output, jpegCodec, encoderParameters);
                    }
                    else
                    {
                        image.Save(output, targetFormat);
                    }
                    
                    return output.ToArray();
                }
            }
        }

        public static byte[] CropImage(byte[] imageData, Rectangle cropArea)
        {
            using (var ms = new MemoryStream(imageData))
            using (var originalImage = new Bitmap(ms))
            {
                using (var croppedImage = originalImage.Clone(cropArea, originalImage.PixelFormat))
                using (var output = new MemoryStream())
                {
                    croppedImage.Save(output, ImageFormat.Png);
                    return output.ToArray();
                }
            }
        }

        public static (int width, int height) GetImageDimensions(byte[] imageData)
        {
            using (var ms = new MemoryStream(imageData))
            using (var image = Image.FromStream(ms))
            {
                return (image.Width, image.Height);
            }
        }

        public static byte[] ApplyGaussianBlur(byte[] imageData, int radius)
        {
            using (var ms = new MemoryStream(imageData))
            using (var bitmap = new Bitmap(ms))
            {
                var blurred = GaussianBlur(bitmap, radius);
                using (var output = new MemoryStream())
                {
                    blurred.Save(output, ImageFormat.Png);
                    return output.ToArray();
                }
            }
        }

        private static Bitmap GaussianBlur(Bitmap source, int radius)
        {
            var result = new Bitmap(source.Width, source.Height);
            
            // Simple box blur as approximation - for production, use proper Gaussian kernel
            using (var graphics = Graphics.FromImage(result))
            {
                graphics.DrawImage(source, 0, 0);
            }
            
            // TODO: Implement proper Gaussian blur
            return result;
        }

        private static ImageCodecInfo GetEncoder(ImageFormat format)
        {
            var codecs = ImageCodecInfo.GetImageDecoders();
            foreach (var codec in codecs)
            {
                if (codec.FormatID == format.Guid)
                {
                    return codec;
                }
            }
            return null;
        }

        public static byte[] ConvertToGrayscale(byte[] imageData)
        {
            using (var ms = new MemoryStream(imageData))
            using (var original = new Bitmap(ms))
            {
                var grayscale = new Bitmap(original.Width, original.Height);
                
                using (var graphics = Graphics.FromImage(grayscale))
                {
                    var colorMatrix = new ColorMatrix(
                        new float[][]
                        {
                            new float[] {.3f, .3f, .3f, 0, 0},
                            new float[] {.59f, .59f, .59f, 0, 0},
                            new float[] {.11f, .11f, .11f, 0, 0},
                            new float[] {0, 0, 0, 1, 0},
                            new float[] {0, 0, 0, 0, 1}
                        });
                    
                    var attributes = new ImageAttributes();
                    attributes.SetColorMatrix(colorMatrix);
                    
                    graphics.DrawImage(original,
                        new Rectangle(0, 0, original.Width, original.Height),
                        0, 0, original.Width, original.Height,
                        GraphicsUnit.Pixel, attributes);
                }
                
                using (var output = new MemoryStream())
                {
                    grayscale.Save(output, ImageFormat.Png);
                    return output.ToArray();
                }
            }
        }
    }
}

