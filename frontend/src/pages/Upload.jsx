import { useState } from "react";
import UploadCT from "../components/UploadCT";
import UploadAudio from "../components/UploadAudio";
import PatientForm from "../components/PatientForm";
import { uploadData } from "../api/backend";
import { useNavigate } from "react-router-dom";

export default function Upload() {
  const [ct, setCT] = useState(null);
  const [audio, setAudio] = useState(null);
  const [metadata, setMetadata] = useState({});
  const navigate = useNavigate();

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("ct_zip", ct);
    formData.append("audio", audio);
    formData.append("metadata", JSON.stringify(metadata));

    const res = await uploadData(formData);
    navigate(`/results/${res.data.id}`);
  };

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Upload Patient Data</h2>

      <UploadCT setCT={setCT} />
      <UploadAudio setAudio={setAudio} />
      <PatientForm metadata={metadata} setMetadata={setMetadata} />

      <button
        onClick={handleUpload}
        className="mt-6 bg-green-600 text-white px-6 py-3 rounded-lg w-full"
      >
        Run Analysis
      </button>
    </div>
  );
}