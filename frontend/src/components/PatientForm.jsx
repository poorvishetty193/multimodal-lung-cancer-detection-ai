export default function PatientForm({ metadata, setMetadata }) {
  return (
    <div className="mt-4 space-y-3">
      <input
        type="number"
        placeholder="Age"
        className="border p-2 w-full"
        onChange={(e) => setMetadata({ ...metadata, age: e.target.value })}
      />

      <select
        className="border p-2 w-full"
        onChange={(e) => setMetadata({ ...metadata, sex: e.target.value })}
      >
        <option value="">Select Sex</option>
        <option value="male">Male</option>
        <option value="female">Female</option>
      </select>
    </div>
  );
}