'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Text } from '@react-three/drei';
import { Suspense, useEffect, useState, useMemo, useRef } from 'react';
import * as THREE from 'three';

interface Point {
  id: string;
  type: 'image' | 'text';
  position: [number, number, number];
  class_id: string;
  class_name: string;
  class_index: number;
  color: string;
  caption: string;
  label: string;
}

interface EmbeddingsData {
  metadata: {
    num_classes: number;
    num_samples: number;
    samples_per_class: number;
    embedding_dim: number;
    classes: Record<string, { name: string; color: string; index: number }>;
  };
  points: Point[];
}

interface PointCloudProps {
  points: Point[];
  type: 'image' | 'text';
  onPointHover: (point: Point | null) => void;
  onPointClick?: (point: Point) => void;
  scale?: number;
}

function PointCloud({ points, type, onPointHover, onPointClick, scale = 5 }: PointCloudProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const colorArray = useRef<Float32Array | null>(null);

  const filteredPoints = useMemo(
    () => points.filter(p => p.type === type), 
    [points, type]
  );

  useEffect(() => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;
    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();

    colorArray.current = new Float32Array(filteredPoints.length * 3);

    filteredPoints.forEach((point, i) => {
      tempObject.position.set(
        point.position[0] * scale,
        point.position[1] * scale,
        point.position[2] * scale
      );
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);

      tempColor.set(point.color);
      colorArray.current![i * 3] = tempColor.r;
      colorArray.current![i * 3 + 1] = tempColor.g;
      colorArray.current![i * 3 + 2] = tempColor.b;
    });

    mesh.instanceMatrix.needsUpdate = true;
    mesh.geometry.setAttribute(
      'color',
      new THREE.InstancedBufferAttribute(colorArray.current, 3)
    );
  }, [filteredPoints, scale]);

  const geometry = useMemo(() => {
    if (type === 'image') {
      return new THREE.SphereGeometry(0.02, 8, 8);
    } else {
      return new THREE.BoxGeometry(0.025, 0.025, 0.025);
    }
  }, [type]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, undefined, filteredPoints.length]}
      onPointerOver={(e) => {
        e.stopPropagation();
        const idx = e.instanceId;
        if (idx !== undefined) {
          onPointHover(filteredPoints[idx]);
          document.body.style.cursor = 'pointer';
        }
      }}
      onPointerOut={() => {
        onPointHover(null);
        document.body.style.cursor = 'default';
      }}
      onClick={(e) => {
        e.stopPropagation();
        const idx = e.instanceId;
        if (idx !== undefined && onPointClick) {
          onPointClick(filteredPoints[idx]);
        }
      }}
    >
      <meshBasicMaterial vertexColors />
    </instancedMesh>
  );
}

function Tooltip({ point }: { point: Point | null }) {
  if (!point) return null;

  return (
    <Html
      position={[
        point.position[0] * 5,
        point.position[1] * 5 + 0.15,
        point.position[2] * 5
      ]}
      center
      style={{ pointerEvents: 'none' }}
    >
      <div className="bg-black/80 backdrop-blur-sm text-white px-3 py-2 rounded-lg text-sm max-w-xs shadow-xl border border-white/10">
        <div className="font-semibold text-xs uppercase tracking-wide opacity-60 mb-1">
          {point.type === 'image' ? 'Image' : 'Text'}
        </div>
        <div className="font-medium">{point.class_name}</div>
        <div className="text-xs opacity-70 mt-1">{point.caption}</div>
      </div>
    </Html>
  );
}

function ClusterLabels({ points, scale = 5 }: { points: Point[]; scale?: number }) {
  const centroids = useMemo(() => {
    const clusters = new Map<string, { positions: number[][]; class_name: string; color: string }>();
    
    points.forEach(point => {
      if (!clusters.has(point.class_id)) {
        clusters.set(point.class_id, { positions: [], class_name: point.class_name, color: point.color });
      }
      clusters.get(point.class_id)!.positions.push(point.position);
    });

    return Array.from(clusters.entries()).map(([class_id, data]) => {
      const avg = data.positions.reduce(
        (acc, pos) => [acc[0] + pos[0], acc[1] + pos[1], acc[2] + pos[2]],
        [0, 0, 0]
      ).map(v => v / data.positions.length);
      
      return {
        class_id,
        class_name: data.class_name,
        color: data.color,
        position: avg as [number, number, number]
      };
    });
  }, [points]);

  return (
    <group>
      {centroids.map((centroid) => (
        <Text
          key={centroid.class_id}
          position={[
            centroid.position[0] * scale,
            centroid.position[1] * scale + 0.2,
            centroid.position[2] * scale
          ]}
          fontSize={0.08}
          color={centroid.color}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.004}
          outlineColor="#000000"
          fillOpacity={0.7}
        >
          {centroid.class_name}
        </Text>
      ))}
    </group>
  );
}

interface ViewSettings {
  autoRotate: boolean;
  showLabels: boolean;
  showGrid: boolean;
  showImages: boolean;
  showText: boolean;
  rotateSpeed: number;
}

interface SceneProps {
  data: EmbeddingsData;
  onImageClick: (point: Point) => void;
  settings: ViewSettings;
}

function Scene({ data, onImageClick, settings }: SceneProps) {
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);

  const handlePointClick = (point: Point) => {
    if (point.type === 'image') {
      onImageClick(point);
    }
  };

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      
      {settings.showImages && (
        <PointCloud 
          points={data.points} 
          type="image" 
          onPointHover={setHoveredPoint}
          onPointClick={handlePointClick}
        />
      )}
      {settings.showText && (
        <PointCloud 
          points={data.points} 
          type="text" 
          onPointHover={setHoveredPoint}
        />
      )}
      
      {settings.showLabels && <ClusterLabels points={data.points} />}
      <Tooltip point={hoveredPoint} />
      
      <OrbitControls 
        enableDamping 
        dampingFactor={0.05}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        panSpeed={0.5}
        minDistance={0.5}
        maxDistance={20}
        autoRotate={settings.autoRotate}
        autoRotateSpeed={settings.rotateSpeed}
      />
      
      {settings.showGrid && (
        <gridHelper args={[10, 20, '#333', '#222']} position={[0, -1, 0]} />
      )}
    </>
  );
}

function LoadingSpinner() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-black">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-white/20 border-t-white rounded-full animate-spin" />
        <p className="text-white/60 text-sm">Loading knowledge space...</p>
      </div>
    </div>
  );
}

function Legend({ classes }: { classes: EmbeddingsData['metadata']['classes'] }) {
  const [isOpen, setIsOpen] = useState(false);
  const uniqueColors = useMemo(() => {
    const colorMap = new Map<string, string[]>();
    Object.values(classes).forEach(c => {
      if (!colorMap.has(c.color)) {
        colorMap.set(c.color, []);
      }
      colorMap.get(c.color)!.push(c.name);
    });
    return colorMap;
  }, [classes]);

  return (
    <div className="absolute bottom-4 left-4 z-10">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-black/60 backdrop-blur-sm text-white px-4 py-2 rounded-lg text-sm border border-white/10 hover:bg-black/80 transition-colors"
      >
        {isOpen ? 'Hide Legend' : 'Show Legend'}
      </button>
      
      {isOpen && (
        <div className="mt-2 bg-black/80 backdrop-blur-sm rounded-lg p-4 max-h-80 overflow-y-auto border border-white/10">
          <div className="text-white/60 text-xs uppercase tracking-wide mb-3">Color Groups</div>
          <div className="space-y-2">
            {Array.from(uniqueColors.entries()).slice(0, 10).map(([color, names]) => (
              <div key={color} className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full flex-shrink-0" 
                  style={{ backgroundColor: color }}
                />
                <span className="text-white/80 text-xs truncate">
                  {names.slice(0, 3).join(', ')}{names.length > 3 ? '...' : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stats({ data }: { data: EmbeddingsData }) {
  const imageCount = useMemo(() => data.points.filter(p => p.type === 'image').length, [data]);
  const textCount = useMemo(() => data.points.filter(p => p.type === 'text').length, [data]);

  return (
    <div className="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-sm rounded-lg p-4 border border-white/10">
      <h2 className="text-white font-semibold text-lg mb-2">CLIP Knowledge Space</h2>
      <div className="space-y-1 text-sm">
        <div className="text-white/70">
          <span className="text-white">{data.metadata.num_classes}</span> classes
        </div>
        <div className="text-white/70">
          <span className="text-white">{imageCount}</span> image embeddings
        </div>
        <div className="text-white/70">
          <span className="text-white">{textCount}</span> text embeddings
        </div>
      </div>
    </div>
  );
}

interface ControlsProps {
  settings: ViewSettings;
  onSettingsChange: (settings: ViewSettings) => void;
}

function Controls({ settings, onSettingsChange }: ControlsProps) {
  const toggle = (key: keyof ViewSettings) => {
    if (typeof settings[key] === 'boolean') {
      onSettingsChange({ ...settings, [key]: !settings[key] });
    }
  };

  return (
    <div className="absolute top-4 right-4 z-10 bg-black/60 backdrop-blur-sm rounded-lg p-4 border border-white/10 w-52">
      <div className="text-white/60 text-xs uppercase tracking-wide mb-3">Controls</div>
      
      <div className="space-y-2 mb-4">
        <div className="text-white/60 text-xs uppercase tracking-wide">View Options</div>
        
        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-white text-xs">Auto Rotate</span>
          <button
            onClick={() => toggle('autoRotate')}
            className={`w-10 h-5 rounded-full transition-colors ${
              settings.autoRotate ? 'bg-blue-500' : 'bg-white/20'
            }`}
          >
            <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
              settings.autoRotate ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-white text-xs">Show Labels</span>
          <button
            onClick={() => toggle('showLabels')}
            className={`w-10 h-5 rounded-full transition-colors ${
              settings.showLabels ? 'bg-blue-500' : 'bg-white/20'
            }`}
          >
            <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
              settings.showLabels ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-white text-xs">Show Grid</span>
          <button
            onClick={() => toggle('showGrid')}
            className={`w-10 h-5 rounded-full transition-colors ${
              settings.showGrid ? 'bg-blue-500' : 'bg-white/20'
            }`}
          >
            <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
              settings.showGrid ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-white text-xs">Show Images</span>
          <button
            onClick={() => toggle('showImages')}
            className={`w-10 h-5 rounded-full transition-colors ${
              settings.showImages ? 'bg-blue-500' : 'bg-white/20'
            }`}
          >
            <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
              settings.showImages ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </button>
        </label>

        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-white text-xs">Show Text</span>
          <button
            onClick={() => toggle('showText')}
            className={`w-10 h-5 rounded-full transition-colors ${
              settings.showText ? 'bg-blue-500' : 'bg-white/20'
            }`}
          >
            <div className={`w-4 h-4 bg-white rounded-full transition-transform mx-0.5 ${
              settings.showText ? 'translate-x-5' : 'translate-x-0'
            }`} />
          </button>
        </label>
      </div>

      {settings.autoRotate && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-white/60 text-xs">Rotation Speed</span>
            <span className="text-white text-xs">{settings.rotateSpeed.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.1"
            value={settings.rotateSpeed}
            onChange={(e) => onSettingsChange({ ...settings, rotateSpeed: parseFloat(e.target.value) })}
            className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      )}

      <div className="border-t border-white/10 pt-3">
        <div className="text-white/60 text-xs uppercase tracking-wide mb-2">Mouse</div>
        <div className="space-y-1 text-xs text-white/70">
          <div><span className="text-white">Left Drag</span> - Rotate</div>
          <div><span className="text-white">Scroll</span> - Zoom</div>
          <div><span className="text-white">Right Drag</span> - Pan</div>
          <div><span className="text-white">Click</span> - Select</div>
        </div>
      </div>
    </div>
  );
}

function ImagePreview({ point, onClose }: { point: Point; onClose: () => void }) {
  const imgIndex = point.id.replace('img_', '');
  const thumbnailSrc = `/thumbnails/img_${imgIndex}.jpg`;

  return (
    <div className="absolute bottom-4 right-4 z-10 bg-black/80 backdrop-blur-sm rounded-lg p-4 border border-white/10 w-64">
      <div className="flex justify-between items-start mb-2">
        <div className="text-white/60 text-xs uppercase tracking-wide">Selected Image</div>
        <button 
          onClick={onClose}
          className="text-white/60 hover:text-white text-sm leading-none"
        >
          x
        </button>
      </div>
      <div className="w-full aspect-square rounded-lg overflow-hidden mb-3 bg-black/50">
        <img 
          src={thumbnailSrc} 
          alt={point.class_name}
          className="w-full h-full object-cover"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = 'none';
          }}
        />
      </div>
      <div className="space-y-1">
        <div className="text-white font-medium text-sm">{point.class_name}</div>
        <div className="text-white/60 text-xs">{point.label}</div>
        <div className="text-white/50 text-xs mt-1">{point.caption}</div>
      </div>
    </div>
  );
}

export default function KnowledgeSpace() {
  const [data, setData] = useState<EmbeddingsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<Point | null>(null);
  const [settings, setSettings] = useState<ViewSettings>({
    autoRotate: false,
    showLabels: true,
    showGrid: true,
    showImages: true,
    showText: true,
    rotateSpeed: 1.0,
  });

  useEffect(() => {
    fetch('/embeddings_data.json')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load data');
        return res.json();
      })
      .then(setData)
      .catch(err => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="w-screen h-screen flex items-center justify-center bg-black text-red-400">
        Error: {error}
      </div>
    );
  }

  if (!data) {
    return <LoadingSpinner />;
  }

  return (
    <div className="w-screen h-screen bg-black relative">
      <Canvas
        camera={{ position: [3, 2, 3], fov: 60 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#0a0a0a']} />
        <fog attach="fog" args={['#0a0a0a', 5, 25]} />
        <Suspense fallback={null}>
          <Scene data={data} onImageClick={setSelectedImage} settings={settings} />
        </Suspense>
      </Canvas>
      
      <Stats data={data} />
      <Controls settings={settings} onSettingsChange={setSettings} />
      <Legend classes={data.metadata.classes} />
      {selectedImage && (
        <ImagePreview point={selectedImage} onClose={() => setSelectedImage(null)} />
      )}
    </div>
  );
}
