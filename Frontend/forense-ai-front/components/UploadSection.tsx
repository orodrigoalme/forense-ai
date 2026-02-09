import React, { useCallback, useRef, useState, useEffect } from 'react';
import { Upload, FileSearch, ShieldCheck, Key, AlertCircle } from 'lucide-react';
import { authService } from '../services/authService';

interface UploadSectionProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
  onKeyChange: () => void;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ onUpload, isLoading, onKeyChange }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [customKey, setCustomKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const savedKey = authService.getCustomKey();
    if (savedKey) {
        setCustomKey(savedKey);
        setShowKeyInput(true);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  }, [onUpload]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
    // Reset value to allow selecting same file again if needed
    if (fileInputRef.current) {
        fileInputRef.current.value = '';
    }
  };

  const handleKeySave = () => {
    if (customKey.trim()) {
        authService.setCustomKey(customKey.trim());
    } else {
        authService.setCustomKey(null);
    }
    onKeyChange(); // Trigger session refresh
  };

  return (
    <div className="w-full max-w-2xl mx-auto my-12 animate-slide-up space-y-6">
        {isLoading ? (
            <div className="flex flex-col items-center justify-center p-12 bg-slate-800/30 rounded-2xl border border-slate-700 text-center">
                 <div className="relative w-20 h-20 mb-6">
                    <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <FileSearch className="absolute inset-0 m-auto text-blue-400 w-8 h-8 animate-pulse" />
                 </div>
                 <h3 className="text-xl font-bold text-white mb-2">Processando Imagem com IA</h3>
                 <p className="text-slate-400 text-sm">Realizando handshake seguro e análise espectral...</p>
                 <div className="mt-4 flex gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0s' }}></span>
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                 </div>
            </div>
        ) : (
            <>
                {/* Upload Box */}
                <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`relative group cursor-pointer border-2 border-dashed rounded-2xl p-12 transition-all duration-300 text-center
                        ${isDragging 
                            ? 'border-blue-500 bg-blue-500/10 scale-[1.02]' 
                            : 'border-slate-700 hover:border-blue-400 hover:bg-slate-800/50 bg-slate-800/30'
                        }
                    `}
                >
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        className="hidden" 
                        accept="image/jpeg,image/png,image/webp"
                        onChange={handleFileSelect}
                    />
                    
                    <div className="flex justify-center mb-6">
                        <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                            <Upload className="w-8 h-8 text-slate-400 group-hover:text-blue-400" />
                        </div>
                    </div>
                    
                    <h3 className="text-xl font-semibold text-slate-200 mb-2">
                        Arraste sua imagem ou clique para analisar
                    </h3>
                    <p className="text-slate-400 text-sm max-w-md mx-auto mb-6">
                        Suporta JPEG, PNG, WEBP. Tamanho máximo 10MB.
                    </p>

                    <div className="flex justify-center gap-6 text-xs text-slate-500">
                        <div className="flex items-center gap-1">
                            <ShieldCheck className="w-4 h-4" />
                            Sessão Anônima Ativa
                        </div>
                        <div className="flex items-center gap-1">
                            <FileSearch className="w-4 h-4" />
                            API v2.0 Live
                        </div>
                    </div>
                </div>

                {/* API Key Toggle Section */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 transition-all">
                    <button 
                        onClick={() => setShowKeyInput(!showKeyInput)}
                        className="flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-blue-400 mb-2 transition-colors w-full text-left"
                    >
                        <Key className="w-3 h-3" />
                        {showKeyInput ? 'Ocultar Configuração Avançada' : 'Tenho uma Chave Gemini (Opcional)'}
                    </button>
                    
                    {showKeyInput && (
                        <div className="animate-fade-in space-y-3">
                            <div className="flex gap-2">
                                <input 
                                    type="password"
                                    value={customKey}
                                    onChange={(e) => setCustomKey(e.target.value)}
                                    placeholder="Cole sua API Key do Google AI Studio aqui..."
                                    className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                                <button 
                                    onClick={handleKeySave}
                                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-lg transition-colors"
                                >
                                    Aplicar
                                </button>
                            </div>
                            <p className="text-[10px] text-slate-500 flex items-start gap-1.5">
                                <AlertCircle className="w-3 h-3 shrink-0 mt-0.5" />
                                Usar sua própria chave aumenta os limites de requisição e quota por sessão. 
                                A chave é enviada via header seguro (X-Gemini-Key) e não é armazenada no servidor.
                            </p>
                        </div>
                    )}
                </div>
            </>
        )}
    </div>
  );
};