import dynamic from 'next/dynamic';

const KnowledgeSpace = dynamic(
  () => import('./components/KnowledgeSpace'),
  // { ssr: false }
);

export default function Home() {
  return <KnowledgeSpace />;
}
