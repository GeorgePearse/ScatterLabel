import React, { useState } from 'react';
import ScatterPlot from './components/ScatterPlot';
import ImageGrid from './components/ImageGrid';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

function App() {
  const [selectedData, setSelectedData] = useState([]);

  const handleClearSelection = () => {
    // Dispatch event to clear scatter plot selection
    window.dispatchEvent(new Event('clearSelection'));
    setSelectedData([]);
  };

  return (
    <div className="App" style={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        backgroundColor: '#f8f9fa',
        borderBottom: '1px solid #dee2e6',
        padding: '10px 20px',
        flexShrink: 0
      }}>
        <h1 style={{ margin: 0, fontSize: '24px', color: '#333' }}>ScatterLabel Viewer</h1>
      </div>

      {/* Split view container */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Scatter plot container */}
        <div style={{
          flex: 1,
          minHeight: '50%',
          borderBottom: '2px solid #dee2e6',
          overflow: 'hidden',
          position: 'relative'
        }}>
          <ErrorBoundary>
            <ScatterPlot onSelectionChange={setSelectedData} />
          </ErrorBoundary>
          
          {/* Clear selection button */}
          {selectedData.length > 0 && (
            <button
              onClick={handleClearSelection}
              style={{
                position: 'absolute',
                bottom: 20,
                right: 20,
                padding: '10px 20px',
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold',
                zIndex: 1000,
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
              }}
            >
              Clear Selection ({selectedData.length})
            </button>
          )}
        </div>

        {/* Image grid container */}
        <div style={{
          flex: 1,
          minHeight: '50%',
          backgroundColor: '#fafafa',
          overflow: 'hidden'
        }}>
          <ErrorBoundary>
            <ImageGrid 
              selectedData={selectedData} 
              imageBasePath="/images" // Adjust this path based on your image server setup
            />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}

export default App;