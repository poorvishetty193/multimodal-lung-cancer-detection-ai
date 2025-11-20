import { useEffect, useState } from "react";
import { getPatients } from "../api/backend";
import { Link } from "react-router-dom";

export default function Patients() {
  const [list, setList] = useState([]);

  useEffect(() => {
    getPatients().then((res) => setList(res.data));
  }, []);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Patient Records</h2>

      {list.map((p) => (
        <Link to={`/results/${p._id}`} key={p._id}>
          <div className="p-4 border rounded mb-3 hover:bg-gray-100">
            <p><b>Age:</b> {p.metadata.age}</p>
            <p><b>Sex:</b> {p.metadata.sex}</p>
            <p><b>Fusion Score:</b> {p.fusion_score}</p>
          </div>
        </Link>
      ))}
    </div>
  );
}