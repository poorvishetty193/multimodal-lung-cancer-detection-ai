import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="p-10 text-center">
      <h1 className="text-4xl font-bold">Lung Cancer Detection AI</h1>
      <p className="mt-4 text-gray-600">CT + Audio + Metadata Fusion</p>

      <Link to="/upload">
        <button className="mt-6 bg-blue-600 text-white px-6 py-3 rounded-lg">
          Start Diagnosis
        </button>
      </Link>
    </div>
  );
}