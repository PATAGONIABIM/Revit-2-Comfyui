// ModernProgressDialog.cs - Un diálogo de progreso moderno y atractivo para WabiSabi Bridge

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Threading;
using System.Threading.Tasks;

namespace WabiSabiBridge.UI
{
    public class ModernProgressDialog : Form
    {
        private Panel progressPanel;
        private Label titleLabel;
        private Label statusLabel;
        private Label percentLabel;
        private CustomProgressBar progressBar;
        private Button cancelButton;
        private PictureBox iconBox;
        private System.Windows.Forms.Timer animationTimer;
        private CancellationTokenSource? cancellationTokenSource;
        
        public ModernProgressDialog(string title = "Procesando...")
        {
            InitializeComponents(title);
            SetupAnimations();
        }

        private void InitializeComponents(string title)
        {
            // Configuración del formulario
            Text = "WabiSabi Bridge";
            Size = new Size(500, 220);
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.None;
            BackColor = Color.FromArgb(245, 245, 245);
            DoubleBuffered = true;
            
            // Panel principal con sombra
            progressPanel = new Panel
            {
                Size = new Size(480, 200),
                Location = new Point(10, 10),
                BackColor = Color.White
            };
            progressPanel.Paint += DrawPanelShadow;
            
            // Icono animado
            iconBox = new PictureBox
            {
                Size = new Size(48, 48),
                Location = new Point(20, 20),
                BackColor = Color.Transparent
            };
            DrawSpinnerIcon();
            
            // Título
            titleLabel = new Label
            {
                Text = title,
                Location = new Point(80, 20),
                Size = new Size(380, 30),
                Font = new Font("Segoe UI", 14F, FontStyle.Bold),
                ForeColor = Color.FromArgb(51, 51, 51),
                BackColor = Color.Transparent
            };
            
            // Estado actual
            statusLabel = new Label
            {
                Text = "Inicializando...",
                Location = new Point(80, 55),
                Size = new Size(380, 25),
                Font = new Font("Segoe UI", 10F),
                ForeColor = Color.FromArgb(102, 102, 102),
                BackColor = Color.Transparent
            };
            
            // Barra de progreso personalizada
            progressBar = new CustomProgressBar
            {
                Location = new Point(30, 95),
                Size = new Size(420, 30),
                BackColor = Color.FromArgb(240, 240, 240),
                ForeColor = Color.FromArgb(0, 120, 215),
                Value = 0
            };
            
            // Porcentaje
            percentLabel = new Label
            {
                Text = "0%",
                Location = new Point(30, 130),
                Size = new Size(100, 20),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 120, 215),
                BackColor = Color.Transparent
            };
            
            // Botón cancelar
            cancelButton = new Button
            {
                Text = "Cancelar",
                Location = new Point(350, 150),
                Size = new Size(100, 30),
                Font = new Font("Segoe UI", 9F),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(240, 240, 240),
                ForeColor = Color.FromArgb(51, 51, 51),
                Cursor = Cursors.Hand
            };
            cancelButton.FlatAppearance.BorderColor = Color.FromArgb(200, 200, 200);
            cancelButton.Click += (s, e) => CancelOperation();
            
            // Agregar controles
            progressPanel.Controls.AddRange(new Control[] {
                iconBox, titleLabel, statusLabel, progressBar, percentLabel, cancelButton
            });
            Controls.Add(progressPanel);
            
            // Eventos para arrastrar la ventana
            MouseDown += DragWindow;
            progressPanel.MouseDown += DragWindow;
            titleLabel.MouseDown += DragWindow;
        }

        private void SetupAnimations()
        {
            animationTimer = new System.Windows.Forms.Timer
            {
                Interval = 50 // 20 FPS
            };
            
            int angle = 0;
            animationTimer.Tick += (s, e) =>
            {
                angle = (angle + 10) % 360;
                RotateIcon(angle);
            };
            animationTimer.Start();
        }

        private void DrawSpinnerIcon()
        {
            var bitmap = new Bitmap(48, 48);
            using (var g = Graphics.FromImage(bitmap))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.Clear(Color.Transparent);
                
                var pen = new Pen(Color.FromArgb(0, 120, 215), 3);
                var rect = new Rectangle(6, 6, 36, 36);
                
                for (int i = 0; i < 8; i++)
                {
                    var alpha = 255 - (i * 30);
                    pen.Color = Color.FromArgb(alpha, 0, 120, 215);
                    g.DrawArc(pen, rect, i * 45, 30);
                }
            }
            iconBox.Image = bitmap;
        }

        private void RotateIcon(int angle)
        {
            if (iconBox.Image == null) return;
            
            var bitmap = new Bitmap(48, 48);
            using (var g = Graphics.FromImage(bitmap))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.TranslateTransform(24, 24);
                g.RotateTransform(angle);
                g.TranslateTransform(-24, -24);
                g.DrawImage(iconBox.Image, 0, 0);
            }
            
            var oldImage = iconBox.Image;
            iconBox.Image = bitmap;
            oldImage?.Dispose();
        }

        private void DrawPanelShadow(object sender, PaintEventArgs e)
        {
            var panel = sender as Panel;
            if (panel == null) return;
            
            using (var path = new GraphicsPath())
            {
                path.AddRectangle(new Rectangle(0, 0, panel.Width, panel.Height));
                
                using (var brush = new PathGradientBrush(path))
                {
                    brush.WrapMode = WrapMode.Clamp;
                    var colorBlend = new ColorBlend(3);
                    colorBlend.Colors = new Color[] {
                        Color.FromArgb(0, Color.Black),
                        Color.FromArgb(20, Color.Black),
                        Color.FromArgb(40, Color.Black)
                    };
                    colorBlend.Positions = new float[] { 0f, 0.95f, 1f };
                    brush.InterpolationColors = colorBlend;
                    
                    e.Graphics.FillRectangle(brush, new Rectangle(-5, -5, panel.Width + 10, panel.Height + 10));
                }
            }
            
            e.Graphics.FillRectangle(new SolidBrush(panel.BackColor), new Rectangle(0, 0, panel.Width, panel.Height));
        }

        private void DragWindow(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                ReleaseCapture();
                SendMessage(Handle, WM_NCLBUTTONDOWN, HT_CAPTION, 0);
            }
        }

        public void UpdateProgress(int percent, string status)
        {
            if (InvokeRequired)
            {
                Invoke(new Action(() => UpdateProgress(percent, status)));
                return;
            }
            
            progressBar.Value = Math.Min(100, Math.Max(0, percent));
            percentLabel.Text = $"{percent}%";
            statusLabel.Text = status;
            
            // Cambiar color según el progreso
            if (percent < 30)
                progressBar.ForeColor = Color.FromArgb(0, 120, 215);
            else if (percent < 70)
                progressBar.ForeColor = Color.FromArgb(0, 150, 0);
            else if (percent < 100)
                progressBar.ForeColor = Color.FromArgb(255, 140, 0);
            else
                progressBar.ForeColor = Color.FromArgb(0, 180, 0);
        }

        public async Task<T> RunAsync<T>(Func<IProgress<(int percent, string status)>, CancellationToken, Task<T>> operation)
        {
            cancellationTokenSource = new CancellationTokenSource();
            var progress = new Progress<(int percent, string status)>(update => UpdateProgress(update.percent, update.status));
            
            try
            {
                Show();
                return await operation(progress, cancellationTokenSource.Token);
            }
            finally
            {
                Close();
                cancellationTokenSource?.Dispose();
            }
        }

        private void CancelOperation()
        {
            if (MessageBox.Show("¿Está seguro de que desea cancelar la operación?", 
                "Confirmar cancelación", 
                MessageBoxButtons.YesNo, 
                MessageBoxIcon.Question) == DialogResult.Yes)
            {
                cancellationTokenSource?.Cancel();
                statusLabel.Text = "Cancelando...";
                cancelButton.Enabled = false;
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            animationTimer?.Stop();
            animationTimer?.Dispose();
            iconBox.Image?.Dispose();
            base.OnFormClosed(e);
        }

        // P/Invoke para arrastrar ventana sin bordes
        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern int SendMessage(IntPtr hWnd, int Msg, int wParam, int lParam);
        
        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool ReleaseCapture();
        
        private const int WM_NCLBUTTONDOWN = 0xA1;
        private const int HT_CAPTION = 0x2;
    }

    // Barra de progreso personalizada con animación suave
    public class CustomProgressBar : UserControl
    {
        private int _value = 0;
        private int _animatedValue = 0;
        private System.Windows.Forms.Timer animationTimer;
        
        public int Value
        {
            get => _value;
            set
            {
                _value = Math.Min(100, Math.Max(0, value));
                Invalidate();
            }
        }
        
        public CustomProgressBar()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | 
                    ControlStyles.UserPaint | 
                    ControlStyles.DoubleBuffer | 
                    ControlStyles.ResizeRedraw, true);
            
            animationTimer = new System.Windows.Forms.Timer { Interval = 16 }; // 60 FPS
            animationTimer.Tick += (s, e) =>
            {
                if (_animatedValue != _value)
                {
                    var diff = _value - _animatedValue;
                    _animatedValue += Math.Max(1, Math.Abs(diff) / 10) * Math.Sign(diff);
                    Invalidate();
                }
            };
            animationTimer.Start();
        }
        
        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            
            // Fondo
            using (var brush = new SolidBrush(BackColor))
            {
                g.FillRoundedRectangle(brush, ClientRectangle, 15);
            }
            
            // Progreso
            if (_animatedValue > 0)
            {
                var progressRect = new Rectangle(0, 0, (int)(Width * _animatedValue / 100.0), Height);
                
                using (var brush = new LinearGradientBrush(progressRect, ForeColor, 
                    Color.FromArgb(255, Math.Min(255, ForeColor.R + 30), 
                                        Math.Min(255, ForeColor.G + 30), 
                                        Math.Min(255, ForeColor.B + 30)), 
                    LinearGradientMode.Horizontal))
                {
                    g.FillRoundedRectangle(brush, progressRect, 15);
                }
                
                // Efecto de brillo
                var highlightRect = new Rectangle(0, 0, progressRect.Width, Height / 2);
                using (var brush = new LinearGradientBrush(highlightRect,
                    Color.FromArgb(80, 255, 255, 255),
                    Color.FromArgb(0, 255, 255, 255),
                    LinearGradientMode.Vertical))
                {
                    g.FillRoundedRectangle(brush, highlightRect, 15);
                }
            }
        }
        
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                animationTimer?.Stop();
                animationTimer?.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    // Extensiones para dibujar rectángulos redondeados
    public static class GraphicsExtensions
    {
        public static void FillRoundedRectangle(this Graphics g, Brush brush, Rectangle rect, int radius)
        {
            using (var path = GetRoundedRectangle(rect, radius))
            {
                g.FillPath(brush, path);
            }
        }
        
        private static GraphicsPath GetRoundedRectangle(Rectangle rect, int radius)
        {
            var path = new GraphicsPath();
            int diameter = radius * 2;
            
            path.AddArc(rect.Left, rect.Top, diameter, diameter, 180, 90);
            path.AddArc(rect.Right - diameter, rect.Top, diameter, diameter, 270, 90);
            path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
            path.AddArc(rect.Left, rect.Bottom - diameter, diameter, diameter, 90, 90);
            path.CloseFigure();
            
            return path;
        }
    }
}
