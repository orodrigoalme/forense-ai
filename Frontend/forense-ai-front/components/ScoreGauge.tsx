import React, { useEffect, useState } from 'react';

interface ScoreGaugeProps {
  score: number; // 0 to 1
  size?: number;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({ score, size = 160 }) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  
  useEffect(() => {
    // Simple animation effect
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 100);
    return () => clearTimeout(timer);
  }, [score]);

  // Determine color based on score (PDF thresholds)
  let color = '#F59E0B'; // Amber default
  if (score <= 0.35) color = '#10B981'; // Green (Real)
  if (score >= 0.55) color = '#EF4444'; // Red (AI)

  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (animatedScore * circumference);

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      {/* Background Circle */}
      <svg className="transform -rotate-90 w-full h-full">
        <circle
          className="text-slate-800"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        {/* Progress Circle */}
        <circle
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      {/* Text Content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold font-mono tracking-tighter" style={{ color }}>
          {(animatedScore * 100).toFixed(0)}%
        </span>
        <span className="text-xs text-slate-500 uppercase tracking-widest mt-1">
          AI Score
        </span>
      </div>
    </div>
  );
};