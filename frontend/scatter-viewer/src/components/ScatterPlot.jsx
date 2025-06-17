import React, { useEffect, useRef, useState } from 'react';
import createScatterplot from 'regl-scatterplot';
import Papa from 'papaparse';

const ScatterPlot = () => {
  const canvasRef = useRef(null);
  const scatterplotRef = useRef(null);
  const [data, setData] = useState(null);
  const [selectedPoints, setSelectedPoints] = useState([]);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  useEffect(() => {
    // Load CSV data
    Papa.parse('/scatterplot_data.csv', {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        console.log('Loaded data:', results.data);
        setData(results.data);
      },
      error: (error) => {
        console.error('Error loading CSV:', error);
      }
    });
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;

    // Extract coordinates and prepare points
    const points = data.map((row, i) => [
      parseFloat(row.tsne_x),
      parseFloat(row.tsne_y)
    ]);

    // Get unique classes for coloring
    const uniqueClasses = [...new Set(data.map(d => d.class_name))];
    const colorScale = createColorScale(uniqueClasses.length);
    
    // Map classes to colors
    const classColorMap = {};
    uniqueClasses.forEach((className, i) => {
      classColorMap[className] = colorScale[i];
    });

    // Create colors array
    const colors = data.map(d => classColorMap[d.class_name]);

    // Initialize scatterplot
    const { width, height } = canvasRef.current.getBoundingClientRect();
    
    const scatterplot = createScatterplot({
      canvas: canvasRef.current,
      width,
      height,
      pointSize: 8,
      pointSizeSelected: 12,
      pointOutlineWidth: 2,
      opacity: 0.8,
      lassoColor: [0, 0.6, 1, 1],
      lassoMinDelay: 15,
      lassoMinDist: 5,
      showReticle: true,
      reticleColor: [1, 1, 1, 0.8]
    });

    scatterplotRef.current = scatterplot;

    // Draw points
    scatterplot.draw(points, { color: colors });

    // Handle point hover
    const handleMouseMove = (event) => {
      try {
        const rect = canvasRef.current.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        const pointIdx = scatterplot.get('closestPoint', { x, y });
        
        if (pointIdx >= 0 && pointIdx < data.length) {
          setHoveredPoint({
            index: pointIdx,
            data: data[pointIdx],
            x: event.clientX,
            y: event.clientY
          });
        } else {
          setHoveredPoint(null);
        }
      } catch (error) {
        console.error('Error in hover handler:', error);
      }
    };

    // Handle selection
    const handleSelect = ({ points: selectedIndices }) => {
      setSelectedPoints(selectedIndices);
      console.log('Selected points:', selectedIndices.map(i => data[i]));
    };

    scatterplot.subscribe('select', handleSelect);
    canvasRef.current.addEventListener('mousemove', handleMouseMove);
    canvasRef.current.addEventListener('mouseleave', () => setHoveredPoint(null));

    // Handle window resize
    const handleResize = () => {
      const { width, height } = canvasRef.current.getBoundingClientRect();
      scatterplot.set({ width, height });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      scatterplot.destroy();
      window.removeEventListener('resize', handleResize);
    };
  }, [data]);

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
      <canvas 
        ref={canvasRef} 
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'fixed',
            left: hoveredPoint.x + 10,
            top: hoveredPoint.y - 40,
            background: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            maxWidth: '300px'
          }}
        >
          <div><strong>Class:</strong> {hoveredPoint.data.class_name}</div>
          <div><strong>ID:</strong> {hoveredPoint.data.annotation_id}</div>
          <div><strong>Confidence:</strong> {hoveredPoint.data.confidence?.toFixed(3)}</div>
          <div><strong>BBox:</strong> [{hoveredPoint.data.x_min}, {hoveredPoint.data.y_min}, {hoveredPoint.data.x_max}, {hoveredPoint.data.y_max}]</div>
        </div>
      )}

      {/* Legend */}
      <ClassLegend data={data} />

      {/* Selection info */}
      {selectedPoints.length > 0 && (
        <div
          style={{
            position: 'absolute',
            bottom: 20,
            left: 20,
            background: 'rgba(255, 255, 255, 0.9)',
            padding: '10px',
            borderRadius: '4px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
        >
          <strong>{selectedPoints.length} points selected</strong>
        </div>
      )}
    </div>
  );
};

// Helper component for class legend
const ClassLegend = ({ data }) => {
  if (!data) return null;

  const uniqueClasses = [...new Set(data.map(d => d.class_name))];
  const colorScale = createColorScale(uniqueClasses.length);

  return (
    <div
      style={{
        position: 'absolute',
        top: 20,
        right: 20,
        background: 'rgba(255, 255, 255, 0.9)',
        padding: '15px',
        borderRadius: '4px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        maxHeight: '400px',
        overflowY: 'auto'
      }}
    >
      <h3 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Classes</h3>
      {uniqueClasses.slice(0, 20).map((className, i) => (
        <div key={className} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
          <div
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: `rgb(${colorScale[i].map(c => Math.round(c * 255)).join(',')})`,
              marginRight: '8px'
            }}
          />
          <span style={{ fontSize: '12px' }}>{className}</span>
        </div>
      ))}
      {uniqueClasses.length > 20 && (
        <div style={{ fontSize: '11px', fontStyle: 'italic', marginTop: '8px' }}>
          ... and {uniqueClasses.length - 20} more
        </div>
      )}
    </div>
  );
};

// Helper function to create color scale
const createColorScale = (n) => {
  const colors = [];
  for (let i = 0; i < n; i++) {
    const hue = (i * 360) / n;
    const rgb = hslToRgb(hue / 360, 0.7, 0.5);
    colors.push(rgb);
  }
  return colors;
};

// HSL to RGB conversion
const hslToRgb = (h, s, l) => {
  let r, g, b;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }

  return [r, g, b, 1];
};

export default ScatterPlot;