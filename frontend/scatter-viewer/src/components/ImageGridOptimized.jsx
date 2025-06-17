import React, { useEffect, useMemo } from 'react';
import useImageLoader from '../hooks/useImageLoader';

const ImageGridOptimized = ({ selectedData, imageBasePath = '/images' }) => {
  const { loadedImages, imageErrors, imageDimensions, loadingImages, loadImages } = useImageLoader(imageBasePath);

  // Group annotations by image_id to avoid duplicate images
  const groupedAnnotations = useMemo(() => {
    const grouped = {};
    selectedData.forEach(point => {
      if (!grouped[point.image_id]) {
        grouped[point.image_id] = [];
      }
      grouped[point.image_id].push(point);
    });
    return grouped;
  }, [selectedData]);

  // Extract unique image IDs
  const imageIds = useMemo(() => 
    [...new Set(selectedData.map(p => p.image_id))],
    [selectedData]
  );

  // Load images when selection changes
  useEffect(() => {
    loadImages(imageIds);
  }, [imageIds, loadImages]);

  if (selectedData.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#666',
        fontSize: '18px'
      }}>
        Select points in the scatter plot to view images
      </div>
    );
  }

  return (
    <div style={{
      padding: '20px',
      height: '100%',
      overflowY: 'auto',
      backgroundColor: '#f5f5f5'
    }}>
      <div style={{
        marginBottom: '20px',
        fontSize: '16px',
        fontWeight: 'bold',
        color: '#333'
      }}>
        {selectedData.length} annotations across {Object.keys(groupedAnnotations).length} images
      </div>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: '20px'
      }}>
        {Object.entries(groupedAnnotations).map(([imageId, annotations]) => (
          <ImageCard
            key={imageId}
            imageId={imageId}
            annotations={annotations}
            imageSrc={loadedImages[imageId]}
            hasError={imageErrors[imageId]}
            isLoading={loadingImages[imageId]}
            dimensions={imageDimensions[imageId]}
          />
        ))}
      </div>
    </div>
  );
};

// Memoized image card component for better performance
const ImageCard = React.memo(({ 
  imageId, 
  annotations, 
  imageSrc, 
  hasError, 
  isLoading,
  dimensions 
}) => {
  return (
    <div
      style={{
        backgroundColor: 'white',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        overflow: 'hidden'
      }}
    >
      <div style={{ position: 'relative' }}>
        {imageSrc && dimensions ? (
          <div style={{ position: 'relative' }}>
            <img
              src={imageSrc}
              alt={`Image ${imageId}`}
              style={{
                width: '100%',
                height: 'auto',
                display: 'block'
              }}
            />
            
            {/* Overlay bounding boxes */}
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none'
              }}
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {annotations.map((ann) => {
                const x = (ann.x_min / dimensions.width) * 100;
                const y = (ann.y_min / dimensions.height) * 100;
                const width = ((ann.x_max - ann.x_min) / dimensions.width) * 100;
                const height = ((ann.y_max - ann.y_min) / dimensions.height) * 100;
                
                return (
                  <g key={ann.annotation_id}>
                    <rect
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill="none"
                      stroke="#00ff00"
                      strokeWidth="0.5"
                      opacity="0.8"
                    />
                    <text
                      x={x}
                      y={y - 1}
                      fill="#00ff00"
                      fontSize="3"
                      fontWeight="bold"
                    >
                      {ann.class_name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        ) : hasError ? (
          <div style={{
            height: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#f0f0f0',
            color: '#666'
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>⚠️</div>
              <div>Image not found</div>
              <div style={{ fontSize: '12px', marginTop: '4px' }}>{imageId}</div>
            </div>
          </div>
        ) : (
          <div style={{
            height: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#f0f0f0'
          }}>
            <div style={{ fontSize: '14px', color: '#666' }}>
              {isLoading ? 'Loading...' : 'Preparing...'}
            </div>
          </div>
        )}
      </div>
      
      <div style={{ padding: '12px' }}>
        <div style={{
          fontSize: '12px',
          color: '#666',
          marginBottom: '8px',
          fontFamily: 'monospace'
        }}>
          {imageId}
        </div>
        <div style={{ fontSize: '14px' }}>
          {annotations.length} annotation{annotations.length > 1 ? 's' : ''}:
          <ul style={{ margin: '4px 0 0 0', paddingLeft: '20px' }}>
            {annotations.map(ann => (
              <li key={ann.annotation_id} style={{ fontSize: '12px', marginBottom: '2px' }}>
                <strong>{ann.class_name}</strong> (conf: {ann.confidence?.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
});

ImageCard.displayName = 'ImageCard';

export default ImageGridOptimized;