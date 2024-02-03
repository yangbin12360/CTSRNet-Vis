import "./index.less";

const TitleBar = (props) => {
  const { title } = props;
  return <div className="title-bar">{title}</div>;
};

export default TitleBar;
