export default function UploadCT({ setCT }) {
  return (
    <div>
      <label className="font-semibold">CT Folder (ZIP):</label>
      <input
        type="file"
        accept=".zip"
        onChange={(e) => setCT(e.target.files[0])}
        className="block mt-2"
      />
    </div>
  );
}