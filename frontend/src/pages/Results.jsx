import { useEffect, useState } from "react";
import { getResult } from "../api/backend";
import { useParams } from "react-router-dom";

export default function Results() {
  const { id } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    getResult(id).then((res) => setData(res.data));
  }, []);

  if (!data) return <p className="p-6">Loading...</p>;

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold mb-4">Diagnosis Report</h2>
      <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">
{data.report}
      </pre>
    </div>
  );
}