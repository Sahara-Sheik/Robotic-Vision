from real_data_visualizer import RealDataHemisphereVisualizer
vis = RealDataHemisphereVisualizer()
vis.load_data_from_csv('sample_camera_data.csv')
vis.plot_hemisphere_heatmap()