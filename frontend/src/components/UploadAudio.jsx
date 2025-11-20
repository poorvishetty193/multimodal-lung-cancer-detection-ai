export default function UploadAudio({ setAudio }) {
  return (
    <div className="mt-4">
      <label className="font-semibold">Audio File (.wav):</label>
      <input
        type="file"
        accept=".wav"
        onChange={(e) => setAudio(e.target.files[0])}
        className="block mt-2"
      />
    </div>
  );
}