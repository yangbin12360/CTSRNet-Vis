export const getDate = (date, gap) => {
  const nDate = new Date(date);
  nDate.setDate(nDate.getDate() + gap);
  let y = nDate.getFullYear();
  let m = nDate.getMonth() + 1;
  let d = nDate.getDate();

  return (
    y.toString() + m.toString().padStart(2, "0") + d.toString().padStart(2, "0")
  );
};

export const getDiffDate = (start, end) => {
  start = new Date(start).getTime();
  end = new Date(end).getTime();

  let diff = Math.abs(start - end);
  let days = Math.floor(diff / (1000 * 3600 * 24));
  return parseInt(days / 6);
};

export const generateContinousArr = (start, end) => {
  return Array.from(new Array(end + 1).keys()).slice(start)
}