// ===== Core/Services/DependencyContainer.cs =====
using System;
using System.Collections.Generic;

namespace WabiSabiRevitBridge.Core.Services
{
    /// <summary>
    /// Simple dependency injection container
    /// </summary>
    public class DependencyContainer
    {
        private static readonly Lazy<DependencyContainer> _instance = 
            new Lazy<DependencyContainer>(() => new DependencyContainer());
        
        private readonly Dictionary<Type, object> _services = new Dictionary<Type, object>();
        private readonly Dictionary<Type, Func<object>> _factories = new Dictionary<Type, Func<object>>();
        
        public static DependencyContainer Instance => _instance.Value;
        
        private DependencyContainer() { }
        
        public void RegisterSingleton<TInterface, TImplementation>() 
            where TImplementation : TInterface, new()
        {
            _services[typeof(TInterface)] = new TImplementation();
        }
        
        public void RegisterSingleton<TInterface>(TInterface instance)
        {
            _services[typeof(TInterface)] = instance;
        }
        
        public void RegisterTransient<TInterface, TImplementation>() 
            where TImplementation : TInterface, new()
        {
            _factories[typeof(TInterface)] = () => new TImplementation();
        }
        
        public void RegisterTransient<TInterface>(Func<TInterface> factory)
        {
            _factories[typeof(TInterface)] = () => factory();
        }
        
        public T Resolve<T>()
        {
            var type = typeof(T);
            
            if (_services.ContainsKey(type))
                return (T)_services[type];
            
            if (_factories.ContainsKey(type))
                return (T)_factories[type]();
            
            throw new InvalidOperationException($"Type {type.Name} not registered");
        }
    }
}