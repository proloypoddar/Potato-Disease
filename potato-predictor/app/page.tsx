"use client";

import { useState, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";

interface Prediction {
  class: string;
  confidence: number;
  suggestion: {
    en: string;
    bn: string;
  };
}

export default function Page() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const res = await axios.post(
        `http://${window.location.hostname}:8000/predict`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPrediction(res.data);
    } catch (err) {
      alert("Prediction failed. Is the FastAPI server running?");
    } finally {
      setLoading(false);
    }
  };

  const speakSuggestion = async (language: "en" | "bn") => {
    if (!prediction || speaking) return;
    
    setSpeaking(true);
    try {
      const text = prediction.suggestion[language];
      const response = await axios.post(
        `http://${window.location.hostname}:8000/text-to-speech`,
        null,
        {
          params: { text, language },
          responseType: 'blob'
        }
      );
      
      const audioBlob = new Blob([response.data], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
      }
    } catch (error) {
      console.error("TTS error:", error);
      alert("Text-to-speech failed. Is the server running?");
    } finally {
      setSpeaking(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-green-50 to-lime-100 text-gray-800 flex flex-col items-center justify-center p-6">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-4xl font-extrabold text-green-900 mb-6"
      >
        🥔 আলু পাতার রোগ সনাক্তকরণ 🌿
      </motion.h1>

      <input
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleImageChange}
        className="mb-4 text-sm font-medium text-gray-700"
      />

      {preview && (
        <motion.img
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          src={preview}
          alt="Preview"
          className="w-64 h-64 object-cover border-4 border-green-500 rounded-lg shadow mb-4"
        />
      )}

      <motion.button
        whileTap={{ scale: 0.95 }}
        onClick={handleUpload}
        disabled={loading || !selectedImage}
        className="bg-green-700 text-white px-6 py-2 rounded-full font-semibold hover:bg-green-800 transition-all duration-200 disabled:bg-gray-400"
      >
        {loading ? "⏳ বিশ্লেষণ চলছে..." : "📤 বিশ্লেষণ করুন"}
      </motion.button>

      {prediction && (
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="mt-6 bg-white p-6 rounded-xl shadow-lg w-full max-w-md"
        >
          <p className="text-xl font-bold text-green-800">🔍 সনাক্তকৃত রোগ: {prediction.class}</p>
          <p className="text-sm text-gray-600 mb-4">নিশ্চয়তা: {prediction.confidence}%</p>
          
          <div className="bg-green-100 p-4 rounded mb-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">💡 পরামর্শ:</h3>
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => speakSuggestion("bn")}
                disabled={speaking}
                className="bg-green-600 text-white text-sm px-3 py-1 rounded-full hover:bg-green-700 transition-colors"
              >
                {speaking ? "🔊 চলছে..." : "🔊 শুনুন"}
              </motion.button>
            </div>
            <p className="text-md text-green-700">{prediction.suggestion.bn}</p>
          </div>
          
          <div className="bg-blue-50 p-4 rounded">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">💡 Suggestion:</h3>
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => speakSuggestion("en")}
                disabled={speaking}
                className="bg-blue-600 text-white text-sm px-3 py-1 rounded-full hover:bg-blue-700 transition-colors"
              >
                {speaking ? "🔊 Playing..." : "🔊 Listen"}
              </motion.button>
            </div>
            <p className="text-md text-blue-700">{prediction.suggestion.en}</p>
          </div>
          
          {/* Hidden audio element for playing TTS */}
          <audio ref={audioRef} onEnded={() => setSpeaking(false)} className="hidden" />
        </motion.div>
      )}
      
      <footer className="mt-8 text-center text-sm text-gray-500">
        <p>© 2025 Developed by: Polok Poddar (Proloy)</p>
      </footer>
    </main>
  );
}
