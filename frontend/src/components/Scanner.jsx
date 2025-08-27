import React, { useState, useRef, useCallback } from 'react';

const Scanner = ({ onItemAdded }) => {
  const [isScanning, setIsScanning] = useState(false);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsScanning(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsScanning(false);
    setDetections([]);
    setScanResult(null);
  }, []);

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      setIsProcessing(true);
      try {
        await processImage(blob);
      } catch (error) {
        console.error('Error processing image:', error);
        alert('Error processing image. Please try again.');
      } finally {
        setIsProcessing(false);
      }
    }, 'image/jpeg', 0.8);
  }, []);

  const processImage = async (blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'scan.jpg');

    try {
      const response = await fetch('http://localhost:5000/api/detection/scan-and-add', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setScanResult(result);
        setDetections(result.added_items || []);
        
        if (result.added_items && result.added_items.length > 0) {
          onItemAdded();
        }
      } else {
        throw new Error('Failed to process image');
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      throw error;
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    try {
      await processImage(file);
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error processing file. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Food Scanner</h2>
        <div className="flex space-x-2">
          {!isScanning ? (
            <button
              onClick={startCamera}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
            >
              📷 Start Camera
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
            >
              🛑 Stop Camera
            </button>
          )}
        </div>
      </div>

      {/* Camera View */}
      {isScanning && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Camera View</h3>
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full max-w-2xl mx-auto rounded-lg"
            />
            <canvas
              ref={canvasRef}
              className="hidden"
            />
            <div className="mt-4 flex justify-center space-x-4">
              <button
                onClick={captureImage}
                disabled={isProcessing}
                className="bg-green-600 text-white px-6 py-3 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing ? 'Processing...' : '📸 Capture & Scan'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* File Upload */}
      {!isScanning && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Image</h3>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              disabled={isProcessing}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed inline-block"
            >
              {isProcessing ? 'Processing...' : '📁 Choose Image'}
            </label>
            <p className="mt-2 text-sm text-gray-500">
              Or drag and drop an image here
            </p>
          </div>
        </div>
      )}

      {/* Scan Results */}
      {scanResult && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Scan Results</h3>
          {detections.length > 0 ? (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <p className="text-green-800 font-medium">
                  ✅ Successfully added {detections.length} item(s) to your pantry!
                </p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {detections.map((item, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">
                        {item.name.toLowerCase().includes('apple') ? '🍎' :
                         item.name.toLowerCase().includes('banana') ? '🍌' :
                         item.name.toLowerCase().includes('milk') ? '🥛' :
                         item.name.toLowerCase().includes('bread') ? '🍞' :
                         item.name.toLowerCase().includes('tomato') ? '🍅' : '🥫'}
                      </span>
                      <div>
                        <h4 className="font-medium text-gray-900">{item.name}</h4>
                        <p className="text-sm text-gray-500">
                          Confidence: {Math.round(item.confidence * 100)}%
                        </p>
                        {item.expiry_date && (
                          <p className="text-sm text-gray-500">
                            Expires: {new Date(item.expiry_date).toLocaleDateString()}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800">
                ⚠️ No food items detected. Try taking a clearer photo or uploading a different image.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-3">📋 How to Use</h3>
        <ul className="space-y-2 text-blue-800">
          <li>• Point your camera at food items or their packaging</li>
          <li>• Make sure the items are well-lit and clearly visible</li>
          <li>• For best results, include expiry dates in the frame</li>
          <li>• The AI will automatically detect items and add them to your pantry</li>
          <li>• You can also upload images from your device</li>
        </ul>
      </div>
    </div>
  );
};

export default Scanner;
