import React, { useState, useEffect } from 'react';
import { UploadSection } from './components/UploadSection';
import { HeroSection } from './components/HeroSection';
import { DetailedAnalysis } from './components/DetailedAnalysis';
import { SessionStatus } from './components/SessionStatus';
import { AnalysisResult, SessionStats } from './types';
import { authService } from './services/authService';
import { apiService } from './services/apiService';
import { ScanSearch, FileText, RotateCcw, AlertTriangle, CloudOff } from 'lucide-react';

export default function App() {
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sessionStats, setSessionStats] = useState<SessionStats | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // Initial Session Load
  useEffect(() => {
    const init = async () => {
      try {
        const stats = await authService.initSession();
        setSessionStats(stats);
      } catch (e) {
        console.error("Failed to init session:", e);
      } finally {
        setIsInitializing(false);
      }
    };
    init();
  }, []);

  const refreshStats = async () => {
    try {
      const stats = await authService.fetchSessionStats();
      setSessionStats(stats);
    } catch (e) {
      console.error("Failed to refresh stats:", e);
    }
  };

  const handleUpload = async (file: File) => {
    setStatus('loading');
    setErrorMessage(null);
    
    try {
      const data = await apiService.analyzeImage(file);
      setResult(data);
      setStatus('success');
      refreshStats(); // Update quotas after analysis
    } catch (error) {
      console.error(error);
      setStatus('error');
      setErrorMessage((error as Error).message || "Ocorreu um erro desconhecido durante a análise.");
    }
  };

  const handleReset = () => {
    setStatus('idle');
    setResult(null);
    setErrorMessage(null);
    refreshStats();
  }

  return (
    <div className="min-h-screen pb-20 bg-slate-900 text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-1.5 rounded-lg shadow-lg shadow-blue-500/20">
                <ScanSearch className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-tight tracking-tight">
                Forense AI <span className="text-blue-500">Viewer</span>
              </h1>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">Advanced Image Diagnostics v2.0</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
             {/* Session Badge */}
             {!isInitializing && <SessionStatus stats={sessionStats} onRefresh={refreshStats} />}
             
             {status === 'success' && (
                 <button 
                    onClick={handleReset}
                    className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-sm text-slate-300 transition-colors"
                 >
                    <RotateCcw className="w-4 h-4" /> Nova Análise
                 </button>
             )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        
        {/* Initialization State */}
        {isInitializing && (
             <div className="flex justify-center items-center py-20 animate-pulse">
                <div className="text-slate-500 text-sm">Estabelecendo conexão segura com Forense API...</div>
             </div>
        )}

        {/* Error State */}
        {status === 'error' && (
            <div className="max-w-2xl mx-auto mb-8 animate-slide-up">
                <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-6 flex items-start gap-4">
                    <CloudOff className="w-6 h-6 text-red-400 shrink-0 mt-1" />
                    <div>
                        <h3 className="text-red-400 font-bold mb-1">Erro na Análise</h3>
                        <p className="text-red-200/80 text-sm mb-4">{errorMessage}</p>
                        <button 
                            onClick={() => setStatus('idle')}
                            className="text-xs bg-red-500/20 hover:bg-red-500/30 text-red-300 px-3 py-2 rounded transition-colors"
                        >
                            Tentar Novamente
                        </button>
                    </div>
                </div>
            </div>
        )}

        {!isInitializing && (status === 'idle' || status === 'loading') && (
          <UploadSection 
            onUpload={handleUpload} 
            isLoading={status === 'loading'} 
            onKeyChange={refreshStats}
          />
        )}

        {status === 'success' && result && (
            <div className="animate-slide-up space-y-6">
              
              {/* Top Hero: Score and Verdict */}
              <HeroSection data={result} />

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Annotated Image */}
                <div className="lg:col-span-2 space-y-6">
                   <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-1">
                      <div className="bg-slate-950 rounded-xl overflow-hidden relative group flex justify-center items-center min-h-[300px]">
                        <img 
                            src={result.annotated_image.startsWith('http') || result.annotated_image.startsWith('data:') ? result.annotated_image : `data:image/png;base64,${result.annotated_image}`}
                            alt="Annotated" 
                            className="max-w-full max-h-[600px] object-contain"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity flex flex-col justify-end p-6 pointer-events-none">
                            <h3 className="text-white font-bold text-lg mb-1">Visualização Anotada</h3>
                            <p className="text-slate-300 text-sm">Heatmap combinando anomalias de ELA e Ruído.</p>
                        </div>
                      </div>
                   </div>
                   
                   {/* Evidence List */}
                   <div className="bg-slate-800/30 rounded-xl border border-slate-700/50 p-6">
                      <h3 className="text-lg font-semibold text-white mb-4">Evidências Principais</h3>
                      <div className="grid gap-3">
                         {result.automated_analysis.key_evidence.map((evidence, i) => (
                             <div key={i} className="flex gap-3 p-3 bg-slate-900/50 rounded-lg border-l-4 border-l-blue-500 border-y border-r border-slate-700/50">
                                 <div className="flex-1 text-sm text-slate-300">
                                     <strong className="text-blue-400 block mb-1 text-xs uppercase tracking-wider">
                                         {evidence.includes(':') ? evidence.split(':')[0] : 'DETECÇÃO'}
                                     </strong>
                                     {evidence.includes(':') ? evidence.split(':')[1] : evidence}
                                 </div>
                             </div>
                         ))}
                      </div>
                   </div>
                </div>

                {/* Right Column: Stats & Breakdown */}
                <div className="space-y-6">
                    {/* Individual Scores */}
                    <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-6">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-5">Scores Individuais</h3>
                        <div className="space-y-4">
                            {[
                                { label: 'FFT (Frequência)', val: result.automated_analysis.individual_scores.fft, color: 'bg-purple-500' },
                                { label: 'Noise (Ruído)', val: result.automated_analysis.individual_scores.noise, color: 'bg-rose-500' },
                                { label: 'ELA (Erro)', val: result.automated_analysis.individual_scores.ela, color: 'bg-orange-500' }
                            ].map((item) => (
                                <div key={item.label}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-slate-300">{item.label}</span>
                                        <span className="text-slate-100 font-mono">{(item.val * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                        <div 
                                            className={`h-full ${item.color} transition-all duration-1000`} 
                                            style={{ width: `${item.val * 100}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Recommendation Card */}
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 p-6 shadow-lg">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Recomendação</h3>
                        <div className="flex items-start gap-3">
                            <div className="p-2 bg-amber-500/10 rounded-lg shrink-0">
                                <AlertTriangle className="w-5 h-5 text-amber-500" />
                            </div>
                            <p className="text-sm text-slate-300 leading-relaxed font-medium">
                                {result.automated_analysis.recommendation}
                            </p>
                        </div>
                    </div>
                </div>
              </div>

              {/* Detailed Analysis Tabs */}
              <DetailedAnalysis data={result} />

            </div>
          )
        }
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-8 mt-12 border-t border-slate-800 text-center">
          <p className="text-slate-600 text-xs">
              © 2024 Forense AI Research. Todas as análises são probabilísticas e requerem validação humana.
          </p>
      </footer>
    </div>
  );
}