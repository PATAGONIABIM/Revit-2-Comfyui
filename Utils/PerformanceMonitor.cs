
// ===== Utils/PerformanceMonitor.cs =====
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace WabiSabiRevitBridge.Utils
{
    /// <summary>
    /// Monitors and tracks performance metrics
    /// </summary>
    public class PerformanceMonitor
    {
        private readonly Dictionary<string, PerformanceMetric> _metrics;
        private readonly object _lock = new object();
        private readonly Timer _cleanupTimer;
        
        public PerformanceMonitor()
        {
            _metrics = new Dictionary<string, PerformanceMetric>();
            
            // Clean up old metrics every minute
            _cleanupTimer = new Timer(_ => CleanupOldMetrics(), null, 
                TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
        }

        public IDisposable StartTimer(string operationName)
        {
            return new TimerScope(this, operationName);
        }

        public void RecordMetric(string name, double value, string unit = "ms")
        {
            lock (_lock)
            {
                if (!_metrics.TryGetValue(name, out var metric))
                {
                    metric = new PerformanceMetric { Name = name, Unit = unit };
                    _metrics[name] = metric;
                }
                
                metric.RecordValue(value);
            }
        }

        public PerformanceMetric GetMetric(string name)
        {
            lock (_lock)
            {
                return _metrics.TryGetValue(name, out var metric) ? metric : null;
            }
        }

        public Dictionary<string, PerformanceMetric> GetAllMetrics()
        {
            lock (_lock)
            {
                return new Dictionary<string, PerformanceMetric>(_metrics);
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                _metrics.Clear();
            }
        }

        private void CleanupOldMetrics()
        {
            lock (_lock)
            {
                var cutoffTime = DateTime.UtcNow.AddHours(-1);
                var keysToRemove = new List<string>();
                
                foreach (var kvp in _metrics)
                {
                    if (kvp.Value.LastUpdated < cutoffTime)
                    {
                        keysToRemove.Add(kvp.Key);
                    }
                }
                
                foreach (var key in keysToRemove)
                {
                    _metrics.Remove(key);
                }
            }
        }

        private class TimerScope : IDisposable
        {
            private readonly PerformanceMonitor _monitor;
            private readonly string _operationName;
            private readonly Stopwatch _stopwatch;
            
            public TimerScope(PerformanceMonitor monitor, string operationName)
            {
                _monitor = monitor;
                _operationName = operationName;
                _stopwatch = Stopwatch.StartNew();
            }
            
            public void Dispose()
            {
                _stopwatch.Stop();
                _monitor.RecordMetric(_operationName, _stopwatch.ElapsedMilliseconds);
            }
        }
    }

    public class PerformanceMetric
    {
        private readonly object _lock = new object();
        private readonly List<double> _values = new List<double>();
        private double _sum;
        private double _min = double.MaxValue;
        private double _max = double.MinValue;
        
        public string Name { get; set; }
        public string Unit { get; set; }
        public DateTime LastUpdated { get; private set; }
        public int Count => _values.Count;
        public double Average => Count > 0 ? _sum / Count : 0;
        public double Min => _min == double.MaxValue ? 0 : _min;
        public double Max => _max == double.MinValue ? 0 : _max;
        
        public void RecordValue(double value)
        {
            lock (_lock)
            {
                _values.Add(value);
                _sum += value;
                _min = Math.Min(_min, value);
                _max = Math.Max(_max, value);
                LastUpdated = DateTime.UtcNow;
                
                // Keep only last 1000 values
                if (_values.Count > 1000)
                {
                    _sum -= _values[0];
                    _values.RemoveAt(0);
                }
            }
        }
        
        public double GetPercentile(double percentile)
        {
            lock (_lock)
            {
                if (_values.Count == 0)
                    return 0;
                
                var sorted = new List<double>(_values);
                sorted.Sort();
                
                var index = (int)Math.Ceiling(percentile / 100.0 * sorted.Count) - 1;
                return sorted[Math.Max(0, Math.Min(index, sorted.Count - 1))];
            }
        }
    }
}
