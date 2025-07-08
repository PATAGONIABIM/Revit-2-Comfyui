<!-- ===== Properties/AssemblyInfo.cs ===== -->
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[assembly: AssemblyTitle("WabiSabi Bridge for Revit")]
[assembly: AssemblyDescription("Real-time bridge between Revit and ComfyUI")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("WabiSabi Development Team")]
[assembly: AssemblyProduct("WabiSabi Bridge")]
[assembly: AssemblyCopyright("Copyright © 2024")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

[assembly: ComVisible(false)]
[assembly: Guid("F4B8C6D2-7E5A-4C89-B9F0-1234567890AB")]

[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

<!-- ===== UI/Themes/MaterialDesignTheme.xaml ===== -->
<ResourceDictionary xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                    xmlns:materialDesign="http://materialdesigninxaml.net/winfx/xaml/themes">
    
    <!-- Material Design Resources -->
    <ResourceDictionary.MergedDictionaries>
        <materialDesign:BundledTheme BaseTheme="Dark" PrimaryColor="Indigo" SecondaryColor="Pink" />
        <ResourceDictionary Source="pack://application:,,,/MaterialDesignThemes.Wpf;component/Themes/MaterialDesignTheme.Defaults.xaml" />
    </ResourceDictionary.MergedDictionaries>

    <!-- Custom Styles -->
    <Style x:Key="WabiSabiCard" TargetType="materialDesign:Card">
        <Setter Property="Padding" Value="16"/>
        <Setter Property="Margin" Value="8"/>
        <Setter Property="materialDesign:ShadowAssist.ShadowDepth" Value="Depth2"/>
    </Style>

    <Style x:Key="WabiSabiButton" TargetType="Button" BasedOn="{StaticResource MaterialDesignRaisedButton}">
        <Setter Property="Height" Value="36"/>
        <Setter Property="Margin" Value="4"/>
        <Setter Property="Padding" Value="16,0"/>
    </Style>

    <Style x:Key="WabiSabiGroupBox" TargetType="GroupBox" BasedOn="{StaticResource MaterialDesignGroupBox}">
        <Setter Property="Margin" Value="0,8"/>
        <Setter Property="Padding" Value="8"/>
    </Style>

    <!-- Color Overrides -->
    <SolidColorBrush x:Key="PrimaryHueLightBrush" Color="#7986CB"/>
    <SolidColorBrush x:Key="PrimaryHueMidBrush" Color="#5C6BC0"/>
    <SolidColorBrush x:Key="PrimaryHueDarkBrush" Color="#3949AB"/>
    
    <!-- Animations -->
    <Storyboard x:Key="FadeIn">
        <DoubleAnimation Storyboard.TargetProperty="Opacity" 
                         From="0" To="1" Duration="0:0:0.3"/>
    </Storyboard>
    
    <Storyboard x:Key="SlideIn">
        <ThicknessAnimation Storyboard.TargetProperty="Margin" 
                            From="20,0,0,0" To="0,0,0,0" 
                            Duration="0:0:0.3">
            <ThicknessAnimation.EasingFunction>
                <CubicEase EasingMode="EaseOut"/>
            </ThicknessAnimation.EasingFunction>
        </ThicknessAnimation>
    </Storyboard>
</ResourceDictionary>
