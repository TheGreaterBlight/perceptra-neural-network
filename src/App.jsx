import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Upload, Camera, ThumbsUp, ThumbsDown } from 'lucide-react';

const App = () => {
  const [epoch, setEpoch] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [weights, setWeights] = useState(null);
  const [learningRate, setLearningRate] = useState(0.1);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageFeatures, setImageFeatures] = useState(null);
  const [imagePrediction, setImagePrediction] = useState(null);
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [userCorrections, setUserCorrections] = useState(0);
  const canvasRef = useRef(null);
  
  const irisData = [
    [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0],
    [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
    [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1],
    [5.5, 2.3, 4.0, 1.3, 1], [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1],
    [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1], [6.6, 2.9, 4.6, 1.3, 1],
    [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1]
  ];

  const extractFeaturesFromImage = (imageData) => {
    const data = imageData.data;
    let totalR = 0, totalG = 0, totalB = 0;
    let totalBrightness = 0;
    let purplePixels = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      totalR += r;
      totalG += g;
      totalB += b;
      
      const brightness = (r + g + b) / 3;
      totalBrightness += brightness;
      
      if (b > r && b > g && b > 100) purplePixels++;
    }
    
    const pixelCount = data.length / 4;
    const avgR = totalR / pixelCount;
    const avgG = totalG / pixelCount;
    const avgB = totalB / pixelCount;
    const avgBrightness = totalBrightness / pixelCount;
    
    const feature1 = (avgB / 255) * 7 + 4;
    const feature2 = (avgBrightness / 255) * 2 + 2;
    const feature3 = (purplePixels / pixelCount) * 5 + 1;
    
    const maxRGB = Math.max(avgR, avgG, avgB);
    const minRGB = Math.min(avgR, avgG, avgB);
    const saturation = maxRGB > 0 ? (maxRGB - minRGB) / maxRGB : 0;
    const feature4 = saturation * 2;
    
    return [feature1, feature2, feature3, feature4];
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setFeedbackMessage('');
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setUploadedImage(img);
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const features = extractFeaturesFromImage(imageData);
        setImageFeatures(features);
        
        if (weights) {
          const normalized = normalizeInput(features);
          const { output } = forward(normalized, weights);
          setImagePrediction(output);
        }
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  // Función para entrenar con feedback del usuario
  const trainWithFeedback = (isCorrect) => {
    if (!imageFeatures || !weights || imagePrediction === null) return;
    
    const normalized = normalizeInput(imageFeatures);
    
    // Si es correcto, no hacemos nada (la red ya predijo bien)
    // Si es incorrecto, entrenamos con la etiqueta opuesta
    let targetLabel;
    
    if (isCorrect) {
      // La predicción fue correcta, reforzamos esa predicción
      targetLabel = imagePrediction < 0.5 ? 0 : 1;
      setFeedbackMessage('✅ ¡Genial! La red aprende de tu confirmación');
    } else {
      // La predicción fue incorrecta, entrenamos con la etiqueta opuesta
      targetLabel = imagePrediction < 0.5 ? 1 : 0;
      setFeedbackMessage('📚 Red corregida. Entrenando con la etiqueta correcta...');
    }
    
    // Entrenar con esta muestra varias veces para reforzar el aprendizaje
    let updatedWeights = { ...weights };
    for (let i = 0; i < 10; i++) {
      const { hiddenOutput, output } = forward(normalized, updatedWeights);
      const error = targetLabel - output;
      
      const outputGradient = error * sigmoidDerivative(output);
      const hiddenGradients = hiddenOutput.map((h, j) => 
        outputGradient * updatedWeights.outputWeights[j] * sigmoidDerivative(h)
      );
      
      updatedWeights.outputWeights = updatedWeights.outputWeights.map((weight, j) => 
        weight + learningRate * outputGradient * hiddenOutput[j]
      );
      updatedWeights.outputBias += learningRate * outputGradient;
      
      updatedWeights.hiddenWeights = updatedWeights.hiddenWeights.map((neuronWeights, j) =>
        neuronWeights.map((weight, k) => 
          weight + learningRate * hiddenGradients[k] * normalized[j]
        )
      );
      updatedWeights.hiddenBias = updatedWeights.hiddenBias.map((bias, j) => 
        bias + learningRate * hiddenGradients[j]
      );
    }
    
    setWeights(updatedWeights);
    setUserCorrections(prev => prev + 1);
    
    // Actualizar predicción
    const { output: newOutput } = forward(normalized, updatedWeights);
    setImagePrediction(newOutput);
    
    setTimeout(() => setFeedbackMessage(''), 3000);
  };

  const normalizeData = (data) => {
    const features = data.map(row => row.slice(0, 4));
    const labels = data.map(row => row[4]);
    
    const means = [0, 0, 0, 0];
    const stds = [0, 0, 0, 0];
    
    for (let i = 0; i < 4; i++) {
      const col = features.map(row => row[i]);
      means[i] = col.reduce((a, b) => a + b) / col.length;
      stds[i] = Math.sqrt(col.map(x => (x - means[i]) ** 2).reduce((a, b) => a + b) / col.length);
    }
    
    const normalized = features.map(row => 
      row.map((val, i) => (val - means[i]) / (stds[i] || 1))
    );
    
    return { features: normalized, labels, means, stds };
  };

  const normalizeInput = (input) => {
    const { means, stds } = normalizeData(irisData);
    return input.map((val, i) => (val - means[i]) / (stds[i] || 1));
  };

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const sigmoidDerivative = (x) => x * (1 - x);

  const initializeWeights = () => {
    return {
      hiddenWeights: Array(4).fill(0).map(() => 
        Array(3).fill(0).map(() => Math.random() * 2 - 1)
      ),
      hiddenBias: Array(3).fill(0).map(() => Math.random() * 2 - 1),
      outputWeights: Array(3).fill(0).map(() => Math.random() * 2 - 1),
      outputBias: Math.random() * 2 - 1
    };
  };

  const forward = (input, w) => {
    const hiddenInput = w.hiddenWeights[0].map((_, j) => {
      return input.reduce((sum, val, i) => sum + val * w.hiddenWeights[i][j], 0) + w.hiddenBias[j];
    });
    const hiddenOutput = hiddenInput.map(sigmoid);
    
    const outputInput = hiddenOutput.reduce((sum, val, i) => 
      sum + val * w.outputWeights[i], 0
    ) + w.outputBias;
    const output = sigmoid(outputInput);
    
    return { hiddenOutput, output };
  };

  const trainStep = (data, w, lr) => {
    const { features, labels } = normalizeData(data);
    let totalLoss = 0;
    
    features.forEach((input, idx) => {
      const target = labels[idx];
      const { hiddenOutput, output } = forward(input, w);
      const error = target - output;
      totalLoss += error ** 2;
      
      const outputGradient = error * sigmoidDerivative(output);
      const hiddenGradients = hiddenOutput.map((h, i) => 
        outputGradient * w.outputWeights[i] * sigmoidDerivative(h)
      );
      
      w.outputWeights = w.outputWeights.map((weight, i) => 
        weight + lr * outputGradient * hiddenOutput[i]
      );
      w.outputBias += lr * outputGradient;
      
      w.hiddenWeights = w.hiddenWeights.map((neuronWeights, i) =>
        neuronWeights.map((weight, j) => 
          weight + lr * hiddenGradients[j] * input[i]
        )
      );
      w.hiddenBias = w.hiddenBias.map((bias, i) => 
        bias + lr * hiddenGradients[i]
      );
    });
    
    return { weights: w, loss: totalLoss / features.length };
  };

  const train = () => {
    if (!weights) {
      setWeights(initializeWeights());
    }
    setIsTraining(true);
  };

  useEffect(() => {
    if (isTraining && epoch < 500) {
      const timer = setTimeout(() => {
        const { weights: newWeights, loss } = trainStep(irisData, weights || initializeWeights(), learningRate);
        setWeights(newWeights);
        setLossHistory(prev => [...prev, { epoch: epoch + 1, loss }]);
        setEpoch(prev => prev + 1);
        
        if ((epoch + 1) % 10 === 0) {
          const { features, labels } = normalizeData(irisData);
          const preds = features.map((input, idx) => {
            const { output } = forward(input, newWeights);
            return {
              x: irisData[idx][0],
              y: irisData[idx][1],
              predicted: output,
              actual: labels[idx]
            };
          });
          setPredictions(preds);
        }

        if (imageFeatures) {
          const normalized = normalizeInput(imageFeatures);
          const { output } = forward(normalized, newWeights);
          setImagePrediction(output);
        }
      }, 20);
      return () => clearTimeout(timer);
    } else if (epoch >= 500) {
      setIsTraining(false);
    }
  }, [isTraining, epoch, weights, learningRate, imageFeatures]);

  const reset = () => {
    setEpoch(0);
    setIsTraining(false);
    setLossHistory([]);
    setPredictions([]);
    setWeights(null);
    setUploadedImage(null);
    setImageFeatures(null);
    setImagePrediction(null);
    setFeedbackMessage('');
    setUserCorrections(0);
  };

  const accuracy = predictions.length > 0 
    ? (predictions.filter(p => Math.round(p.predicted) === p.actual).length / predictions.length * 100).toFixed(1)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Red Neuronal Multicapa con Visión</h1>
          <p className="text-gray-600 mb-6">Clasificador de Flores Iris con Aprendizaje Supervisado Interactivo</p>
          
          <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-6 text-white mb-8">
            <h2 className="text-2xl font-bold mb-4">Arquitectura de la Red</h2>
            <div className="flex items-center justify-around">
              <div className="text-center">
                <div className="text-5xl mb-2">🔢</div>
                <div className="font-semibold">Capa de Entrada</div>
                <div className="text-sm">4 neuronas</div>
                <div className="text-xs opacity-80">(características)</div>
              </div>
              <div className="text-4xl">→</div>
              <div className="text-center">
                <div className="text-5xl mb-2">🧠</div>
                <div className="font-semibold">Capa Oculta</div>
                <div className="text-sm">3 neuronas</div>
                <div className="text-xs opacity-80">(sigmoid)</div>
              </div>
              <div className="text-4xl">→</div>
              <div className="text-center">
                <div className="text-5xl mb-2">🎯</div>
                <div className="font-semibold">Capa de Salida</div>
                <div className="text-sm">1 neurona</div>
                <div className="text-xs opacity-80">(sigmoid)</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Época Actual</div>
              <div className="text-3xl font-bold text-blue-600">{epoch}</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Precisión</div>
              <div className="text-3xl font-bold text-green-600">{accuracy}%</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Error Actual</div>
              <div className="text-3xl font-bold text-purple-600">
                {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss.toFixed(4) : '0.0000'}
              </div>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Correcciones</div>
              <div className="text-3xl font-bold text-orange-600">{userCorrections}</div>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tasa de Aprendizaje: {learningRate}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              disabled={isTraining}
              className="w-full"
            />
          </div>

          <div className="flex gap-4 mb-8">
            <button
              onClick={train}
              disabled={isTraining}
              className="flex-1 bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isTraining ? '🔄 Entrenando...' : '🚀 Iniciar Entrenamiento'}
            </button>
            <button
              onClick={reset}
              className="flex-1 bg-gray-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-600 transition-all"
            >
              🔄 Reiniciar
            </button>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <Camera className="w-6 h-6" />
            Clasificar tu Imagen de Flor
          </h2>
          
          {!weights || epoch < 100 ? (
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
              <p className="text-yellow-800">
                ⚠️ Primero entrena la red (al menos 100 épocas) para poder clasificar imágenes
              </p>
            </div>
          ) : null}
          
          <div className="mb-6">
            <label className="flex items-center justify-center w-full h-32 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-lg appearance-none cursor-pointer hover:border-blue-400 focus:outline-none">
              <span className="flex items-center space-x-2">
                <Upload className="w-6 h-6 text-gray-600" />
                <span className="font-medium text-gray-600">
                  Haz clic para cargar una imagen de flor
                </span>
              </span>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                disabled={!weights || epoch < 100}
              />
            </label>
          </div>

          {uploadedImage && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-700 mb-2">Imagen Cargada</h3>
                <img 
                  src={uploadedImage.src} 
                  alt="Flor cargada" 
                  className="w-full rounded-lg shadow-lg"
                  style={{ maxHeight: '300px', objectFit: 'contain' }}
                />
              </div>
              
              <div>
                <h3 className="font-semibold text-gray-700 mb-4">Resultado de Clasificación</h3>
                
                {imageFeatures && (
                  <div className="bg-gray-50 p-4 rounded-lg mb-4">
                    <h4 className="font-medium text-gray-700 mb-2">Características Extraídas:</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Color Azul: {imageFeatures[0].toFixed(2)}</div>
                      <div>Brillo: {imageFeatures[1].toFixed(2)}</div>
                      <div>Púrpura: {imageFeatures[2].toFixed(2)}</div>
                      <div>Saturación: {imageFeatures[3].toFixed(2)}</div>
                    </div>
                  </div>
                )}
                
                {imagePrediction !== null && (
                  <>
                    <div className={`p-6 rounded-lg mb-4 ${imagePrediction < 0.5 ? 'bg-blue-100' : 'bg-red-100'}`}>
                      <div className="text-center">
                        <div className="text-6xl mb-4">
                          {imagePrediction < 0.5 ? '🌸' : '🌺'}
                        </div>
                        <h3 className="text-2xl font-bold mb-2">
                          {imagePrediction < 0.5 ? 'Iris Setosa' : 'Iris Versicolor'}
                        </h3>
                        <div className="text-sm text-gray-600">
                          Confianza: {(Math.abs(imagePrediction - 0.5) * 200).toFixed(1)}%
                        </div>
                        <div className="mt-4 text-xs text-gray-500">
                          Valor de salida: {imagePrediction.toFixed(4)}
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gradient-to-r from-green-50 to-red-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-gray-800 mb-3 text-center">
                        ¿La predicción es correcta?
                      </h4>
                      <div className="flex gap-3">
                        <button
                          onClick={() => trainWithFeedback(true)}
                          className="flex-1 bg-green-500 hover:bg-green-600 text-white px-4 py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all"
                        >
                          <ThumbsUp className="w-5 h-5" />
                          Correcto
                        </button>
                        <button
                          onClick={() => trainWithFeedback(false)}
                          className="flex-1 bg-red-500 hover:bg-red-600 text-white px-4 py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all"
                        >
                          <ThumbsDown className="w-5 h-5" />
                          Incorrecto
                        </button>
                      </div>
                      
                      {feedbackMessage && (
                        <div className="mt-3 text-center text-sm font-medium text-gray-700 bg-white p-2 rounded">
                          {feedbackMessage}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl shadow-2xl p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-4">📉 Curva de Pérdida</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Época', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Error', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="loss" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-2xl shadow-2xl p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-4">🌸 Predicciones Dataset</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" name="Longitud Sépalo" />
                <YAxis dataKey="y" name="Ancho Sépalo" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter 
                  name="Setosa" 
                  data={predictions.filter(p => p.predicted < 0.5)} 
                  fill="#3b82f6" 
                />
                <Scatter 
                  name="Versicolor" 
                  data={predictions.filter(p => p.predicted >= 0.5)} 
                  fill="#ef4444" 
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-6 mt-8">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">📝 Detalles de Implementación</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-2">✅ Características de la Red:</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li>Red multicapa (4→3→1)</li>
                <li>Función Sigmoid en ambas capas</li>
                <li>Backpropagation completo</li>
                <li>Normalización de datos</li>
                <li>Entrenamiento con gradiente descendente</li>
                <li>Aprendizaje supervisado interactivo</li>
              </ul>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-2">🖼️ Procesamiento de Imágenes:</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li>Extracción de características de color</li>
                <li>Análisis de brillo y saturación</li>
                <li>Detección de tonalidades específicas</li>
                <li>Conversión a 4 características numéricas</li>
                <li>Clasificación en tiempo real</li>
                <li>Corrección manual con feedback</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;